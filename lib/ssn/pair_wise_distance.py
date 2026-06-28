import torch
import torch.nn.functional as F
import math

class PairwiseDistFunction:
    @staticmethod
    def apply(pixel_features, spixel_features, init_spixel_indices, num_spixels_w, num_spixels_h):
        """
        Pure PyTorch implementation supporting autograd. 
        Resolves C++ compilation issues and dimension compatibility.
        Args:
            pixel_features: (B, C, H, W) or (B, C, N_pixels)
            spixel_features: (B, C, N_spixels)
            init_spixel_indices: (B, 1, H, W) or (B, H, W) or (B, N_pixels)
            num_spixels_w, num_spixels_h: int
        """
        # 1. Intelligent Dimension Inference and Standardization
        # ---------------------------------------------------------
        if pixel_features.dim() == 3: # (B, C, N) -> needs to be reshaped to (B, C, H, W)
            b, c, n = pixel_features.shape
            
            # Attempt to infer spatial dimensions H and W from indices
            if init_spixel_indices.dim() == 4: # (B, 1, H, W)
                h, w = init_spixel_indices.shape[2], init_spixel_indices.shape[3]
            elif init_spixel_indices.dim() == 3: # (B, H, W)
                h, w = init_spixel_indices.shape[1], init_spixel_indices.shape[2]
            else: # (B, N) or (H, W) - most error-prone case
                # If indices shape is (B, N) and N matches the pixel count
                if init_spixel_indices.shape[1] == n:
                    # Assume a square patch (commonly used during training)
                    h = int(math.sqrt(n))
                    w = n // h
                # If indices shape is (H, W) and batch size b=1
                elif init_spixel_indices.dim() == 2 and b == 1:
                    h, w = init_spixel_indices.shape
                else:
                    # Fallback option: calculate square root directly
                    h = int(math.sqrt(n))
                    w = n // h
            
            # Perform reshaping
            pixel_features = pixel_features.view(b, c, h, w)
            # Restore indices to spatial structure for subsequent coordinate calculations
            if init_spixel_indices.dim() == 2:
                init_spixel_indices = init_spixel_indices.view(b, h, w)
        
        # At this point, pixel_features is guaranteed to be (B, C, H, W)
        b, c, h, w = pixel_features.shape
        device = pixel_features.device

        # Ensure indices shape is (B, H, W)
        if init_spixel_indices.dim() == 4:
            init_spixel_indices = init_spixel_indices.squeeze(1)
        
        init_spixel_indices = init_spixel_indices.long()

        # 2. Prepare 3x3 (9-grid) Neighborhood Offsets
        # ---------------------------------------------------------
        shifts_y = torch.tensor([-1, -1, -1,  0, 0, 0,  1, 1, 1], device=device)
        shifts_x = torch.tensor([-1,  0,  1, -1, 0, 1, -1, 0, 1], device=device)

        # 3. Calculate Grid Coordinates
        # ---------------------------------------------------------
        # Convert absolute IDs from init_spixel_indices to (row, col) coordinates
        sp_y = torch.div(init_spixel_indices, num_spixels_w, rounding_mode='floor') # (B, H, W)
        sp_x = init_spixel_indices % num_spixels_w                                  # (B, H, W)

        # 4. Traverse the 9 Neighbors
        # ---------------------------------------------------------
        dist_list = []
        
        # Preprocess spixel_features for gathering
        # (B, C, N_sp) -> (B, N_sp, C)
        sp_feat_perm = spixel_features.permute(0, 2, 1) 

        for k in range(9):
            # Calculate neighbor coordinates
            neighbor_sp_y = sp_y + shifts_y[k]
            neighbor_sp_x = sp_x + shifts_x[k]

            # Out-of-bounds mask
            valid_mask = (neighbor_sp_y >= 0) & (neighbor_sp_y < num_spixels_h) & \
                         (neighbor_sp_x >= 0) & (neighbor_sp_x < num_spixels_w)

            # Coordinate clamping (prevents indexing out-of-bounds errors)
            safe_y = neighbor_sp_y.clamp(0, num_spixels_h - 1)
            safe_x = neighbor_sp_x.clamp(0, num_spixels_w - 1)

            # Calculate neighbor IDs
            neighbor_id = safe_y * num_spixels_w + safe_x # (B, H, W)

            # Gather features
            # Expand neighbor_id to (B, H, W, C)
            neighbor_id_expanded = neighbor_id.unsqueeze(-1).expand(-1, -1, -1, c)
            
            # Retrieve features: (B, N_sp, C) gather by (B, H, W, C) -> (B, H, W, C)
            # Note: PyTorch gather requires consistent dimensions, so we map the N_sp dimension to H*W.
            # A cleaner approach would be advanced indexing, but gather is more robust inside modules.
            # Flatten the spatial dimensions first: (B, H*W, C)
            # sp_feat_perm is (B, N_sp, C)
            # This is essentially a lookup table operation.
            
            # Proper usage of gather:
            # dim=1 is the lookup dimension.
            # neighbor_id_flat = neighbor_id.view(b, h*w, 1).expand(-1, -1, c)
            # neighbor_feat = torch.gather(sp_feat_perm, 1, neighbor_id_flat) # (B, H*W, C)
            # neighbor_feat = neighbor_feat.view(b, h, w, c)
            
            neighbor_id_flat = neighbor_id.view(b, h*w, 1).expand(-1, -1, c)
            neighbor_feat = torch.gather(sp_feat_perm, 1, neighbor_id_flat)
            neighbor_feat = neighbor_feat.view(b, h, w, c).permute(0, 3, 1, 2) # (B, C, H, W)

            # Calculate distance
            diff = pixel_features - neighbor_feat
            dist_sq = torch.sum(diff * diff, dim=1) # (B, H, W)

            # Handle out-of-bounds coordinates: set distance to 1e6
            dist_sq = torch.where(valid_mask, dist_sq, torch.tensor(1e6, device=device))

            dist_list.append(dist_sq)

        # 5. Output Results
        # ---------------------------------------------------------
        # Shape: (B, 9, H, W)
        output = torch.stack(dist_list, dim=1)

        # Reshape back to flattened format (B, 9, N) to match downstream SSN operations
        output = output.view(b, 9, -1)

        return output
