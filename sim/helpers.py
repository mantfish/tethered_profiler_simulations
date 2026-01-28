import numpy as np

def edit_dat_file(template_path, new_path, replacements):
    # replacements is a dictionary {old_value: new_value}
    with open(template_path, 'r') as file:
        content = file.read()

    for old_val, new_val in replacements.items():
        content = content.replace(old_val, new_val)

    with open(new_path, 'w') as file:
        file.write(content)


def get_linear_current_at_depth(z, current_speed, depth):
    # map z in [-depth, 0] -> alpha in [0, 1]
    alpha = (z + depth) / depth
    alpha = np.clip(alpha, 0.0, 1.0)
    return current_speed * alpha