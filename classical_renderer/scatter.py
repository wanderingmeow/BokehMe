#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

import numpy as np
import mlx.core as mx

def render_update_output(imageMx, defocus):
    assert imageMx.ndim == 4, "`image` must be 4D."
    assert defocus.ndim == 4, "`defocus` must be 4D."

    assert imageMx.shape[2:4] == defocus.shape[2:4], "`image` and `defocus` must have same resolution"

    # n, channels, h, w = image.shape

    source = """
    uint elem = thread_position_in_grid.x;

    int N = image_shape[0];
    int C = image_shape[1];
    int H = image_shape[2];
    int W = image_shape[3];

    int n = (elem / C / W / H) % N;
    int y = (elem / C / W) % H;
    int x = (elem / C) % W;

    T fltDefocus = defocus[n * W * H * C + 0 * W * H + y * W + x];
    T fltRadius = metal::fabs(fltDefocus);

    for (int intDeltaY = -int(fltRadius) - 1; intDeltaY <= int(fltRadius) + 1; ++intDeltaY) {
        for (int intDeltaX = -int(fltRadius) - 1; intDeltaX <= int(fltRadius) + 1; ++intDeltaX) {
            int intNeighborY = y + intDeltaY;
            int intNeighborX = x + intDeltaX;

            if ((intNeighborY >= 0) && (intNeighborY < H) && (intNeighborX >= 0) && (intNeighborX < W)) {
                T fltDist = metal::sqrt(T(intDeltaY) * T(intDeltaY) + T(intDeltaX) * T(intDeltaX));
                T fltWeight = (0.5 + 0.5 * metal::tanh(4 * (fltRadius - fltDist))) / (fltRadius * fltRadius + 0.2);

                if (fltRadius >= fltDist) {
                    atomic_fetch_max_explicit(&defocusDilateMax[intNeighborY * W + intNeighborX], int(fltDefocus), memory_order_relaxed);
                }

                atomic_fetch_add_explicit(&weightCum[W * intNeighborY + intNeighborX], fltWeight, memory_order_relaxed);
                atomic_fetch_add_explicit(&bokehCum[n * C * W * H + 0 * W * H + intNeighborY * W + intNeighborX], fltWeight * image[n * C * W * H + 0 * W * H + y * W + x], memory_order_relaxed);
                atomic_fetch_add_explicit(&bokehCum[n * C * W * H + 1 * W * H + intNeighborY * W + intNeighborX], fltWeight * image[n * C * W * H + 1 * W * H + y * W + x], memory_order_relaxed);
                atomic_fetch_add_explicit(&bokehCum[n * C * W * H + 2 * W * H + intNeighborY * W + intNeighborX], fltWeight * image[n * C * W * H + 2 * W * H + y * W + x], memory_order_relaxed);
            }
        }
    }
    """

    kernel = mx.fast.metal_kernel(
        name="render",
        input_names=[
            "image",    # original image
            "defocus",  # signed defocus map
        ],
        output_names=[
            "defocusDilateMax", # signed defocus map after dilating
            "bokehCum",         # cumulative bokeh image
            "weightCum",        # cumulative weight map
        ],
        source=source,
        atomic_outputs=True,
    )

    imageMx = mx.array(imageMx)
    defocusMx = mx.array(defocus)

    outputs = kernel(
        inputs=[imageMx, defocusMx],
        output_shapes=[defocusMx.shape, imageMx.shape, defocusMx.shape],
        output_dtypes=[mx.int32, imageMx.dtype, defocusMx.dtype],
        grid=(np.prod(imageMx.shape), 1, 1),
        threadgroup=(256, 1, 1),
        template=[("T", mx.float32)],
        init_value=0,
    )

    defocusDilateMax = np.array(outputs[0], copy=False)
    defocusDilate = torch.from_numpy(np.maximum(defocus, defocusDilateMax)).int()

    bokehCum = torch.from_numpy(np.array(outputs[1], copy=False))
    weightCum = torch.from_numpy(np.array(outputs[2], copy=False))

    return defocusDilate, bokehCum, weightCum


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, image, defocus):
        # self.save_for_backward(image, defocus)

        if not mx.metal.is_available():
            raise NotImplementedError()

        return render_update_output(image.numpy(), defocus.numpy())

        # end

    # end

    # @staticmethod
    # def backward(self, gradBokehCum, gradWeightCum):
    # end

# end


def FunctionRender(image, defocus):
    defocus_dilate, bokeh_cum, weight_cum = _FunctionRender.apply(image, defocus)

    return defocus_dilate, bokeh_cum, weight_cum
# end


class ModuleRenderScatter(torch.nn.Module):
    def __init__(self):
        super(ModuleRenderScatter, self).__init__()
    # end

    def forward(self, image, defocus):
        defocus_dilate, bokeh_cum, weight_cum = FunctionRender(image, defocus)
        bokeh = bokeh_cum / weight_cum
        return bokeh, defocus_dilate
    # end
# end
