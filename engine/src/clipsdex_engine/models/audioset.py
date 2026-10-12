"""AudioSet facts the audio tagger uses. Kept free of torch so importing it is instant."""

SAMPLE_RATE = 16_000  # what the Cnn14_16k tagger expects
NUM_CLASSES = 527

# Class indices, verified against AudioSet's class_labels_indices.csv (2026-10-12).
SPEECH = 0
MUSIC = 137
# Singing, Male/Female/Child/Synthetic singing, Rapping, Vocal music.
VOCAL_CLASSES = (27, 32, 33, 34, 35, 36, 254)
