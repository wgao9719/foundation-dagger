"""Quick test to verify camera binning is symmetric."""
import numpy as np

camera_max_angle = 5.0
n_camera_bins = 5

# Compute bin edges
bin_edges = np.linspace(-camera_max_angle, camera_max_angle, n_camera_bins + 1)
print(f"Full bin edges: {bin_edges}")
print(f"Digitize edges (bin_edges[1:-1]): {bin_edges[1:-1]}")

# Bin centers for decoding
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
print(f"Bin centers: {bin_centers}")
print()

# Test encoding
test_values = np.array([-5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5])
clipped = np.clip(test_values, -camera_max_angle, camera_max_angle)
bins = np.digitize(clipped, bin_edges[1:-1])

print("Encoding test:")
print(f"{'Raw':>8} {'Clipped':>8} {'Bin':>4} {'Center':>8} {'Decoded':>8} {'Error':>8}")
print("-" * 56)
for raw, clip, b in zip(test_values, clipped, bins):
    center = bin_centers[b]
    error = center - raw
    print(f"{raw:8.2f} {clip:8.2f} {b:4d} {center:8.2f} {center:8.2f} {error:+8.2f}")

print()
print("Check for asymmetry:")
# Test symmetric pairs
pairs = [(-4, 4), (-3, 3), (-2, 2), (-1, 1), (-0.5, 0.5)]
for neg, pos in pairs:
    neg_bin = np.digitize(np.clip(neg, -5, 5), bin_edges[1:-1])
    pos_bin = np.digitize(np.clip(pos, -5, 5), bin_edges[1:-1])
    neg_decoded = bin_centers[neg_bin]
    pos_decoded = bin_centers[pos_bin]
    neg_err = neg_decoded - neg
    pos_err = pos_decoded - pos
    print(f"  {neg:+.1f}° → bin {neg_bin} → {neg_decoded:+.1f}° (err {neg_err:+.2f})")
    print(f"  {pos:+.1f}° → bin {pos_bin} → {pos_decoded:+.1f}° (err {pos_err:+.2f})")
    if abs(neg_err) != abs(pos_err):
        print(f"  ⚠️  ASYMMETRIC! neg_err={neg_err:.2f}, pos_err={pos_err:.2f}")
    print()

