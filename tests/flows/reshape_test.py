import unittest
import torch

from antsnormflows.flows import Radial
from tests.flows.flow_test import FlowTest
from antsnormflows.flows import Split

class ReshapeTest(FlowTest):

    def validate_checkerboard_split(self,
                                    input_tensor, 
                                    mode='checkerboard', 
                                    do_visualization=False):

        def create_checkerboard_mask(z_size, inverse=False):
            n_dims = len(z_size)
            if n_dims == 4:  # NCHW
                b, c, h, w = z_size
                row_idx = torch.arange(h).view(1, h, 1)
                col_idx = torch.arange(w).view(1, 1, w)
                mask = (row_idx + col_idx) % 2 == 0
                mask = mask.expand(b, c, -1, -1)
            elif n_dims == 5:  # NCDHW
                b, c, d, h, w = z_size
                depth_idx = torch.arange(d).view(1, d, 1, 1)
                row_idx = torch.arange(h).view(1, 1, h, 1)
                col_idx = torch.arange(w).view(1, 1, 1, w)
                mask = (depth_idx + row_idx + col_idx) % 2 == 0
                mask = mask.expand(b, c, -1, -1, -1)
            else:
                raise ValueError("Input must be 4D (NCHW) or 5D (NCDHW) for checkerboard visualization.")

            return ~mask if inverse else mask

        split_layer = Split(mode=mode)

        # Forward pass
        z, _ = split_layer.forward(input_tensor)
        z1, z2 = z

        # Inverse pass
        output_tensor, _ = split_layer.inverse(z)

        if input_tensor.ndim == 4 and input_tensor.size(1) == 1:
            mask = create_checkerboard_mask(input_tensor.shape, inverse=('inv' in mode))
            masked_z1 = input_tensor.clone()
            masked_z1[~mask] = 0
            masked_z2 = input_tensor.clone()
            masked_z2[mask] = 0
            if do_visualization:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(input_tensor[0, 0].cpu().numpy(), cmap='gray')
                axes[0].set_title("Input")
                axes[0].axis('off')
                axes[1].imshow(masked_z1[0, 0].cpu().numpy(), cmap='gray')
                axes[1].set_title("z1 (Masked)")
                axes[1].axis('off')
                axes[2].imshow(masked_z2[0, 0].cpu().numpy(), cmap='gray')
                axes[2].set_title("z2 (Masked)")
                axes[2].axis('off')
                plt.tight_layout()
                plt.show()
        elif input_tensor.ndim == 5 and input_tensor.size(1) == 1:
            # For 3D, we can visualize a single slice (e.g., the middle depth slice)
            if do_visualization:
                import matplotlib.pyplot as plt

                mid_depth = input_tensor.size(2) // 2
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(input_tensor[0, 0, mid_depth].cpu().numpy(), cmap='gray')
                axes[0].set_title(f"Input (Depth {mid_depth})")
                axes[0].axis('off')
            mask = create_checkerboard_mask(input_tensor.shape, inverse=('inv' in mode))
            masked_z1 = input_tensor.clone()
            masked_z1[~mask] = 0
            masked_z2 = input_tensor.clone()
            masked_z2[mask] = 0
            if do_visualization:
                axes[1].imshow(masked_z1[0, 0, mid_depth].cpu().numpy(), cmap='gray')
                axes[1].set_title(f"z1 (Masked, Depth {mid_depth})")
                axes[1].axis('off')
                axes[2].imshow(masked_z2[0, 0, mid_depth].cpu().numpy(), cmap='gray')
                axes[2].set_title(f"z2 (Masked, Depth {mid_depth})")
                axes[2].axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print("\nVisualization is only available for single-channel 2D or 3D input.")


    def test_checkerboard_split_2d(self):
        
        batch_size = 1
        channels = 1
        height = 8
        width = 8
        do_visualization=False

        input = torch.arange(height * width).float().view(batch_size, channels, height, width)
        self.validate_checkerboard_split(input.clone(), mode='checkerboard', do_visualization=do_visualization)
        self.validate_checkerboard_split(input.clone(), mode='checkerboard_inv', do_visualization=do_visualization)

    def test_checkerboard_split_3d(self):
        
        batch_size = 1
        channels = 1
        depth = 32
        height = 32
        width = 32
        do_visualization=False

        input = torch.arange(depth * height * width).float().view(batch_size, channels, depth, height, width)
        self.validate_checkerboard_split(input.clone(), mode='checkerboard', do_visualization=do_visualization)
        self.validate_checkerboard_split(input.clone(), mode='checkerboard_inv', do_visualization=do_visualization)

    def test_channel_split(self):

        batch_size = 2
        channels = 4
        height = 4
        width = 4

        input = torch.arange(batch_size * channels * height * width).float().view(batch_size, channels, height, width)
        split_layer_channel = Split(mode='channel')
        z_channel, _ = split_layer_channel.forward(input)
        z1_channel, z2_channel = z_channel
        assert z1_channel.shape == torch.Size([2, 2, 4, 4])
        assert z2_channel.shape == torch.Size([2, 2, 4, 4])
        output_channel, _ = split_layer_channel.inverse(z_channel)
        assert output_channel.shape == torch.Size([2, 4, 4, 4])

if __name__ == "__main__":
    unittest.main()
