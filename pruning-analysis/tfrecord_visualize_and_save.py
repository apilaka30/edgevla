# ==============================================================================
# Visualize and Save Samples from TFRecord
# ==============================================================================

import tensorflow as tf
from PIL import Image
import os

# === CONFIG ===
TFRECORD_PATH_LIST = [
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0/ucsd_kitchen_dataset_converted_externally_to_rlds-train.tfrecord-00000-of-00001",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/bridge_oxe/0.1.0/bridge-test.tfrecord-00027-of-00032",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/bridge_oxe/0.1.0/bridge-test.tfrecord-00012-of-00032",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/fractal20220817_data/0.1.0/fractal20220817_data-train.tfrecord-00746-of-01024",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/fractal20220817_data/0.1.0/fractal20220817_data-train.tfrecord-00278-of-01024",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0/nyu_franka_play_dataset_converted_externally_to_rlds-val.tfrecord-00003-of-00032",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0/nyu_franka_play_dataset_converted_externally_to_rlds-val.tfrecord-00022-of-00032",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/berkeley_cable_routing/0.1.0/berkeley_cable_routing-test.tfrecord-00002-of-00004",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/taco_play/0.1.0/taco_play-test.tfrecord-00090-of-00128",
    "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/taco_play/0.1.0/taco_play-test.tfrecord-00020-of-00128"
]
SAVE_DIR = "/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples"
IMG_DIR = os.path.join(SAVE_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# === Helper to parse a TFRecord ===
def parse_example(serialized_example):
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())
    return example

# === Main loop ===
instruction_lines = []
sample_count = 0

for tfrecord_path in TFRECORD_PATH_LIST:
    language_instr_key = "steps/language_instruction" if ("ucsb" in tfrecord_path or "nyu" in tfrecord_path) else "steps/observation/natural_language_instruction"
    img_key = "steps/observation/rgb_static" if ("taco_play" in tfrecord_path) else "steps/observation/image"
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for i, raw_record in enumerate(raw_dataset.take(20) if "cable_routing" in tfrecord_path else raw_dataset.take(80)):
        example = parse_example(raw_record)

        # Extract language instruction
        lang_feature = example.features.feature[language_instr_key].bytes_list.value
        if not lang_feature:
            continue  # Skip if no language instruction
        instruction = lang_feature[0].decode("utf-8")

        # Extract image
        image_feature = example.features.feature[img_key].bytes_list.value
        if not image_feature:
            continue  # Skip if no image
        image_bytes = image_feature[0]
        image = tf.image.decode_jpeg(image_bytes).numpy()

        # Save image
        sample_count += 1
        img_filename = f"sample_img{sample_count}.png"
        img_path = os.path.join(IMG_DIR, img_filename)
        Image.fromarray(image).save(img_path)

        # Save instruction for this sample
        instruction_lines.append(instruction)

# === Write all instructions to file ===
instr_path = os.path.join(SAVE_DIR, "sample_instructions.txt")
with open(instr_path, "w") as f:
    for line in instruction_lines:
        f.write(line.strip() + "\n")

print(f"✅ Saved {sample_count} samples in {SAVE_DIR}")

# ==============================================================================
# Visualize TFRecord Key Structure
# ==============================================================================


# # 🧩 Setup
# import tensorflow as tf
# from PIL import Image

# # ✅ Change this path to your local .tfrecord file
# TFRECORD_PATH = "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/libero_object_no_noops/1.0.0/libero_object-train.tfrecord-00001-of-00032"

# # Utility: Decode byte string to dict if needed
# def try_decode_bytes(value):
#     try:
#         return value.numpy().decode('utf-8')
#     except Exception:
#         return value.numpy()

# # Utility: Parse a single tf.train.Example
# def parse_example(serialized_example):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example.numpy())
#     return example

# # Load the first example
# raw_dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
# first_example = next(iter(raw_dataset))
# parsed_example = parse_example(first_example)

# # 🧠 View available keys (top-level feature dict)
# print("Available keys:", list(parsed_example.features.feature.keys()))



# ==============================================================================
# Example of parsing TFRecord as SequenceExample or Example
# ==============================================================================

# import tensorflow as tf

# tfrecord_path = "/bigscratch/apilaka/rlds_datasets/open_x_embodiment/libero_object_no_noops/1.0.0/libero_object-train.tfrecord-00000-of-00032"
# dataset = tf.data.TFRecordDataset([tfrecord_path])

# raw = next(iter(dataset))

# # Try parsing as SequenceExample
# seq_ex = tf.train.SequenceExample()
# try:
#     seq_ex.ParseFromString(raw.numpy())
#     if seq_ex.feature_lists.feature_list:
#         print("=== Parsed as SequenceExample ===")
#         for key, fl in seq_ex.feature_lists.feature_list.items():
#             first = fl.feature[0]
#             if first.HasField("float_list"):
#                 print(f"{key}: float dim = {len(first.float_list.value)}")
#             elif first.HasField("bytes_list"):
#                 print(f"{key}: bytes len = {len(first.bytes_list.value[0])}")
#             elif first.HasField("int64_list"):
#                 print(f"{key}: int64 len = {len(first.int64_list.value)}")
#     else:
#         print("No feature_lists found.")
# except Exception as e:
#     print("Failed parsing as SequenceExample:", e)

# # Try parsing as Example
# ex = tf.train.Example()
# try:
#     ex.ParseFromString(raw.numpy())
#     if ex.features.feature:
#         print("\n=== Parsed as Example ===")
#         for key, f in ex.features.feature.items():
#             if f.HasField("float_list"):
#                 print(f"{key}: float dim = {len(f.float_list.value)}")
#             elif f.HasField("bytes_list"):
#                 print(f"{key}: bytes len = {len(f.bytes_list.value[0])}")
#             elif f.HasField("int64_list"):
#                 print(f"{key}: int64 dim = {len(f.int64_list.value)}")
#     else:
#         print("No features found.")
# except Exception as e:
#     print("Failed parsing as Example:", e)
