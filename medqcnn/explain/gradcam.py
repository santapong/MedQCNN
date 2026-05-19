"""
Grad-CAM (Selvaraju et al., 2017) for HybridQCNN.

Generates a spatial saliency map by combining activations from the last
convolutional block of the frozen ResNet backbone with the gradient of
the chosen class logit. Despite the quantum layer in the middle, the
end-to-end pipeline is fully differentiable, so autograd carries the
gradient all the way back to the backbone feature map — no special
quantum-aware treatment is needed for this view of the model.

Per HQCM-EBTC (arXiv:2506.21937), pairing classical Grad-CAM with a
hybrid quantum head is a published recipe for medical-image
classification interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class GradCAM:
    """Compute a Grad-CAM heatmap for a HybridQCNN-style model.

    Args:
        model: The trained :class:`~medqcnn.model.hybrid.HybridQCNN`
            (or any nn.Module with a ``.backbone.features`` Sequential).
        target_layer: The conv/block whose activations + gradients drive
            the heatmap. Defaults to the second-to-last child of
            ``backbone.features`` — for ResNet-18 wrapped per
            :class:`medqcnn.classical.backbone.ClassicalBackbone` this is
            ``layer4``.

    The class registers forward/backward hooks on creation; call
    :py:meth:`close` (or use it as a context manager) to remove them.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        if target_layer is None:
            target_layer = self._default_target_layer(model)
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _default_target_layer(model: nn.Module) -> nn.Module:
        backbone = getattr(model, "backbone", None)
        features = getattr(backbone, "features", None)
        if features is None or len(features) < 2:
            raise AttributeError(
                "Could not infer a default Grad-CAM target layer. Pass "
                "target_layer explicitly."
            )
        # ResNet-18 wrapped via list(children())[:-1] places layer4 at -2
        # (the final child is AdaptiveAvgPool2d).
        return features[-2]

    def _save_activation(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
    ) -> torch.Tensor:
        """Return a (H, W) heatmap in [0, 1] for the first sample of ``x``.

        Args:
            x: (1, C, H, W) input batch. Only the first item is used.
            target_class: Class index to attribute. If None, the
                argmax-class is used.
        """
        if x.shape[0] != 1:
            raise ValueError(f"GradCAM expects a single-sample batch, got {x.shape}")

        was_training = self.model.training
        self.model.eval()
        # Re-enable autograd through frozen backbone params so the
        # gradient reaches the activation; the backbone weights are not
        # touched (we never call optimizer.step).
        x = x.clone().detach().requires_grad_(True)

        try:
            self.model.zero_grad()
            logits = self.model(x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]
            score.backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError("Forward/backward hooks did not fire.")

            activations = self._activations[0]  # (C, h, w)
            gradients = self._gradients[0]      # (C, h, w)
            weights = gradients.mean(dim=(1, 2))  # (C,)

            cam = torch.einsum("c,chw->hw", weights, activations)
            cam = functional.relu(cam)
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min < 1e-12:
                return torch.zeros_like(cam)
            cam = (cam - cam_min) / (cam_max - cam_min)

            h, w = x.shape[-2:]
            cam = functional.interpolate(
                cam[None, None, :, :],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            return cam.detach()
        finally:
            if was_training:
                self.model.train()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
