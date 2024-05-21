import tensorflow as tf
from src.data_processing import create_generators
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_history, save_history

# Konfiguracja GPU dla TensorFlow na procesorach M1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Ścieżki do danych treningowych i testowych
train_dir = 'data/train'
test_dir = 'data/test'

# Ustawienia rozmiaru obrazów i batch size
image_size = (224, 224)
batch_size = 32
epochs = 10

# Tworzenie generatorów danych
train_generator, test_generator = create_generators(train_dir, test_dir, image_size, batch_size)

# Tworzenie modelu
model = create_model(image_size)

# Trenowanie modelu
history = train_model(model, train_generator, test_generator, epochs)

# Ewaluacja modelu
evaluate_model(model, test_generator)

# Zapisywanie historii treningu
save_history(history)

# Tworzenie wykresów historii treningu
plot_history(history)