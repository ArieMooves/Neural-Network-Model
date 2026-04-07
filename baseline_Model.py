import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def preprocess_image(index, dataset):
    img = dataset[index]                      
    img = img.astype("float32") / 255.0      
    img_array = np.expand_dims(img, axis=0)  
    return img, img_array


class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def get_class_label(preds):
    return class_names[np.argmax(preds)]


def compute_gradcam(model, img_array, class_index, conv_layer_name=None):
    if conv_layer_name is None:
        conv_layer_name = get_last_conv_layer(model)
       
    # Reach directly into layers to avoid Sequential 'lazy' errors
    model_input = model.layers[0].input
    model_output = model.layers[-1].output
    target_layer_output = model.get_layer(conv_layer_name).output
   
    grad_model = tf.keras.models.Model(
        inputs=model_input,
        outputs=[target_layer_output, model_output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]


    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()


    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]


    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


    overlay = heatmap * alpha + (img * 255).astype(np.uint8) * (1 - alpha)
    return np.uint8(overlay)


def visualize_feature_maps(model, img_array, layer_name):
    """Shows what the neurons in a specific layer 'see' for a given image."""
    # Create a model that outputs the activation of the target layer
    feature_map_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    feature_maps = feature_map_model.predict(img_array)


    # Plot the first 8-16 filters from the layer
    n_features = min(feature_maps.shape[-1], 16)
    size = feature_maps.shape[1]
    display_grid = np.zeros((size, size * n_features))


    for i in range(n_features):
        x = feature_maps[0, :, :, i]
        x -= x.mean(); x /= (x.std() + 1e-5); x *= 64; x += 128
        x = np.clip(x, 0, 255).astype('uint8')
        display_grid[:, i * size : (i + 1) * size] = x


    plt.figure(figsize=(15, 3))
    plt.title(f"Feature Maps for Layer: {layer_name}")
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.show()


def visualize_filters(model, layer_name, iterations=30, learning_rate=10.0):
    """Generates an image that maximizes the activation of a specific filter."""
    layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # Maximize the mean of the activation for that specific filter
    return tf.reduce_mean(activation[:, :, :, filter_index])


    # Visualize the first 4 filters
    plt.figure(figsize=(12, 3))
    for filter_index in range(4):
        # Start with a random noise image
        input_image = tf.random.uniform((1, 32, 32, 3))


        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(input_image)
                loss = compute_loss(input_image, filter_index)
            grads = tape.gradient(loss, input_image)
            grads = tf.math.l2_normalize(grads)
            input_image += learning_rate * grads


        # De-process the image for plotting
        img = input_image.numpy()[0]
        img = (img - img.mean()) / (img.std() + 1e-5) * 0.1 + 0.5
        img = np.clip(img, 0, 1)


        plt.subplot(1, 4, filter_index + 1)
        plt.imshow(img)
        plt.title(f"Filter {filter_index}")
        plt.axis('off')
    plt.suptitle(f"Learned Visual Patterns for {layer_name}")
    plt.show()


if __name__ == "__main__":
    # Load CIFAR-10 properly
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Define Architecture
    # BASELINE MODEL
  model = keras.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),


    # Block 2
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),


    # Block 3
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),


    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")  
  ])

    # Compile and Train (Crucial for meaningful Grad-CAM)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
   
    print("Training model for 5 epochs...")
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


    # Evaluate on CIFAR-10 test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_acc}")


    # Grad-CAM Analysis
    image_index = 7135
    img, img_array = preprocess_image(image_index, x_test)


    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    predicted_label = get_class_label(preds)
    true_label = class_names[y_test[image_index][0]]


    print(f"Predicted: {predicted_label} | Actual: {true_label}")


    heatmap = compute_gradcam(model, img_array, class_index)
    output_img = overlay_heatmap(img, heatmap)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1); plt.imshow(img); plt.title(f"True: {true_label}"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(output_img); plt.title(f"Pred: {predicted_label}"); plt.axis('off')
    plt.show()
    # Feature Map Visualization
    _ = model.predict(np.zeros((1,32,32,3)), verbose=0)
    visualize_feature_maps(model, img_array, model.layers[0].name)
    # Filter Visualization (Pattern Recognition)
    last_conv = get_last_conv_layer(model)
    visualize_filters(model, last_conv)
