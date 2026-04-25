"""
FairSlice — Main Pipeline
Connects vision → partition → visualization
"""

from vision import segment_dish
from partition import compute_partition
from visualize import render_partition
from preferences_ui import build_preference_matrix


def process_dish(image_path: str, n_people: int, mode: str = "free"):
    """
    Full pipeline: photo in → fair partition out.
    """
    # Step 1: Segment ingredients
    ingredient_map, labels = segment_dish(image_path)

    # Step 2: Build per-person preferences matrix from UI inputs.
    # Today we default to uniform sliders; a real UI can replace `user_inputs`.
    K = len(labels)
    user_inputs = {i: [1.0] * K for i in range(n_people)}
    P = build_preference_matrix(n_people, labels, user_inputs)

    # Step 3: Compute fair partition (partition will normalize preferences internally)
    result = compute_partition(ingredient_map, n_people, mode=mode, preferences=P)

    # Step 4: Render output
    output_image = render_partition(
        image_path=image_path,
        masks=result["masks"],
        scores=result["scores"],
        ingredient_labels=labels,
        fairness=result["fairness"]
    )

    return output_image, result["scores"], result["fairness"], labels


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample_images/pizza_test.jpg"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    img, scores, fairness, labels = process_dish(path, n)
    print(f"Fairness score: {fairness:.3f}")
    for i in range(len(scores)):
        print(f"Person {i+1}: {', '.join(f'{labels[k]}={scores[i,k]:.1%}' for k in range(len(labels)))}")
    img.save("output.png")
    print("Saved to output.png")