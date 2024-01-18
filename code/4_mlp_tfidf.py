{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c8536797-0a1a-4425-8e22-ef2e29ee556f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:00:39.319534Z",
     "iopub.status.busy": "2023-05-16T19:00:39.318723Z",
     "iopub.status.idle": "2023-05-16T19:00:39.326012Z",
     "shell.execute_reply": "2023-05-16T19:00:39.323879Z",
     "shell.execute_reply.started": "2023-05-16T19:00:39.319469Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# NB requires 190 GB RAM to run!"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "66076417-10a4-497d-b2e8-b493b6a227ad",
   "metadata": {},
   "source": [
    "### Install dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5b1e071e-9c5f-408d-b013-73d2a1ac8c2e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:30:06.046600Z",
     "iopub.status.busy": "2023-05-17T12:30:06.045909Z",
     "iopub.status.idle": "2023-05-17T12:31:16.218825Z",
     "shell.execute_reply": "2023-05-17T12:31:16.215952Z",
     "shell.execute_reply.started": "2023-05-17T12:30:06.046529Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install tensorflow scikit-learn pandas matplotlib prettytable --quiet"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "cec506e7-af9d-4922-a394-a12d8d65a395",
   "metadata": {},
   "source": [
    "### Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ace51d06",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:31:16.235169Z",
     "iopub.status.busy": "2023-05-17T12:31:16.234869Z",
     "iopub.status.idle": "2023-05-17T12:31:18.878810Z",
     "shell.execute_reply": "2023-05-17T12:31:18.878125Z",
     "shell.execute_reply.started": "2023-05-17T12:31:16.235142Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import time\n",
    "\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import regularizers\n",
    "from keras.utils.np_utils import to_categorical\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.utils import to_categorical\n"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "44d84da8-c927-40b7-8f55-04932ad0cad8",
   "metadata": {},
   "source": [
    "### Load and split the dataset w. TF-IDF embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66f1344a-f523-4992-b0d4-7c07bacff2cf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:31:18.881356Z",
     "iopub.status.busy": "2023-05-17T12:31:18.880752Z",
     "iopub.status.idle": "2023-05-17T12:31:22.809355Z",
     "shell.execute_reply": "2023-05-17T12:31:22.808522Z",
     "shell.execute_reply.started": "2023-05-17T12:31:18.881333Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load sparse tf-idf data\n",
    "X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse, org_train, org_test = pd.read_pickle(\"pickles/sparse.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7ce84a15-1eab-49ab-8bc4-9bac4a5b9389",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:31:22.810493Z",
     "iopub.status.busy": "2023-05-17T12:31:22.810283Z",
     "iopub.status.idle": "2023-05-17T12:31:22.876055Z",
     "shell.execute_reply": "2023-05-17T12:31:22.875165Z",
     "shell.execute_reply.started": "2023-05-17T12:31:22.810473Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Split to train and validation sets\n",
    "X_train_sparse, X_val_sparse, y_train_sparse, y_val_sparse = train_test_split(X_train_sparse, y_train_sparse, test_size=0.2, random_state=42)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "853b4010-40f6-478b-b0cc-887a5a6bcde5",
   "metadata": {},
   "source": [
    "### Scaling and encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cb20b4a6-bf4d-4e14-9d4e-50ebf8cf2c59",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:32:51.731170Z",
     "iopub.status.busy": "2023-05-17T12:32:51.730361Z",
     "iopub.status.idle": "2023-05-17T12:32:52.280471Z",
     "shell.execute_reply": "2023-05-17T12:32:52.279666Z",
     "shell.execute_reply.started": "2023-05-17T12:32:51.731109Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# One-hot encode labels \n",
    "encoder = LabelBinarizer()\n",
    "y_train_enc = encoder.fit_transform(y_train_sparse)\n",
    "y_val_enc = encoder.transform(y_val_sparse)\n",
    "\n",
    "# Transform the test set labels using the existing LabelBinarizer\n",
    "y_test_enc = encoder.transform(y_test_sparse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "337301ff-703d-4fba-9a88-d9e7c9f3ce28",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:32:52.282103Z",
     "iopub.status.busy": "2023-05-17T12:32:52.281794Z",
     "iopub.status.idle": "2023-05-17T12:32:52.554841Z",
     "shell.execute_reply": "2023-05-17T12:32:52.553951Z",
     "shell.execute_reply.started": "2023-05-17T12:32:52.282080Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Scale the TFiDF feature vectors\n",
    "scaler = StandardScaler(with_mean = False)\n",
    "X_train_scaled = scaler.fit_transform(X_train_sparse)\n",
    "X_val_scaled = scaler.transform(X_val_sparse)\n",
    "X_test_scaled = scaler.transform(X_test_sparse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "93e8d586-752b-4c73-8798-9141701dcc34",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T12:32:54.391526Z",
     "iopub.status.busy": "2023-05-17T12:32:54.390803Z",
     "iopub.status.idle": "2023-05-17T12:34:05.158385Z",
     "shell.execute_reply": "2023-05-17T12:34:05.156887Z",
     "shell.execute_reply.started": "2023-05-17T12:32:54.391469Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Convert X train and X validation csr_matrix into tensors\n",
    "X_train_tensor = tf.convert_to_tensor(X_train_scaled.toarray())\n",
    "X_val_tensor = tf.convert_to_tensor(X_val_scaled.toarray())"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c33bf7cf-31cd-418c-b592-6f9740bee22b",
   "metadata": {},
   "source": [
    "### Build and train the TF-IDF MLP classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "428c4e0b-1b2d-4956-b244-94346ca6bfc5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T13:27:44.549809Z",
     "iopub.status.busy": "2023-05-17T13:27:44.548934Z",
     "iopub.status.idle": "2023-05-17T13:58:35.270323Z",
     "shell.execute_reply": "2023-05-17T13:58:35.268437Z",
     "shell.execute_reply.started": "2023-05-17T13:27:44.549748Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "3839/3839 [==============================] - 46s 12ms/step - loss: 2.9213 - accuracy: 0.5546 - val_loss: 2.9496 - val_accuracy: 0.5977\n",
      "Epoch 2/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.8102 - accuracy: 0.5995 - val_loss: 2.8712 - val_accuracy: 0.5901\n",
      "Epoch 3/100\n",
      "3839/3839 [==============================] - 48s 12ms/step - loss: 2.7799 - accuracy: 0.5974 - val_loss: 2.8313 - val_accuracy: 0.5932\n",
      "Epoch 4/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.7045 - accuracy: 0.6035 - val_loss: 2.7562 - val_accuracy: 0.5945\n",
      "Epoch 5/100\n",
      "3839/3839 [==============================] - 44s 12ms/step - loss: 2.6318 - accuracy: 0.6046 - val_loss: 2.6556 - val_accuracy: 0.5997\n",
      "Epoch 6/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.5705 - accuracy: 0.6077 - val_loss: 2.5709 - val_accuracy: 0.5990\n",
      "Epoch 7/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.5034 - accuracy: 0.6102 - val_loss: 2.5290 - val_accuracy: 0.6017\n",
      "Epoch 8/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.4666 - accuracy: 0.6097 - val_loss: 2.5180 - val_accuracy: 0.6055\n",
      "Epoch 9/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.4164 - accuracy: 0.6111 - val_loss: 2.4543 - val_accuracy: 0.6072\n",
      "Epoch 10/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.3914 - accuracy: 0.6113 - val_loss: 2.4625 - val_accuracy: 0.6069\n",
      "Epoch 11/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.3469 - accuracy: 0.6148 - val_loss: 2.3751 - val_accuracy: 0.6086\n",
      "Epoch 12/100\n",
      "3839/3839 [==============================] - 49s 13ms/step - loss: 2.3339 - accuracy: 0.6132 - val_loss: 2.3638 - val_accuracy: 0.6084\n",
      "Epoch 13/100\n",
      "3839/3839 [==============================] - 44s 12ms/step - loss: 2.3324 - accuracy: 0.6141 - val_loss: 2.4095 - val_accuracy: 0.6019\n",
      "Epoch 14/100\n",
      "3839/3839 [==============================] - 44s 11ms/step - loss: 2.3015 - accuracy: 0.6125 - val_loss: 2.3426 - val_accuracy: 0.6076\n",
      "Epoch 15/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2748 - accuracy: 0.6151 - val_loss: 2.3419 - val_accuracy: 0.6087\n",
      "Epoch 16/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2739 - accuracy: 0.6133 - val_loss: 2.3015 - val_accuracy: 0.6075\n",
      "Epoch 17/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2588 - accuracy: 0.6144 - val_loss: 2.3389 - val_accuracy: 0.6093\n",
      "Epoch 18/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2480 - accuracy: 0.6143 - val_loss: 2.2886 - val_accuracy: 0.6076\n",
      "Epoch 19/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2345 - accuracy: 0.6149 - val_loss: 2.3056 - val_accuracy: 0.6052\n",
      "Epoch 20/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2367 - accuracy: 0.6148 - val_loss: 2.3043 - val_accuracy: 0.6063\n",
      "Epoch 21/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2294 - accuracy: 0.6156 - val_loss: 2.2833 - val_accuracy: 0.6095\n",
      "Epoch 22/100\n",
      "3839/3839 [==============================] - 53s 14ms/step - loss: 2.2231 - accuracy: 0.6149 - val_loss: 2.2549 - val_accuracy: 0.6130\n",
      "Epoch 23/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2145 - accuracy: 0.6148 - val_loss: 2.2756 - val_accuracy: 0.6057\n",
      "Epoch 24/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.2087 - accuracy: 0.6154 - val_loss: 2.2540 - val_accuracy: 0.6092\n",
      "Epoch 25/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1975 - accuracy: 0.6146 - val_loss: 2.2427 - val_accuracy: 0.6098\n",
      "Epoch 26/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1913 - accuracy: 0.6153 - val_loss: 2.2775 - val_accuracy: 0.6097\n",
      "Epoch 27/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1949 - accuracy: 0.6167 - val_loss: 2.2681 - val_accuracy: 0.6062\n",
      "Epoch 28/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1992 - accuracy: 0.6146 - val_loss: 2.2248 - val_accuracy: 0.6133\n",
      "Epoch 29/100\n",
      "3839/3839 [==============================] - 44s 12ms/step - loss: 2.2036 - accuracy: 0.6140 - val_loss: 2.2665 - val_accuracy: 0.6075\n",
      "Epoch 30/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1907 - accuracy: 0.6152 - val_loss: 2.2390 - val_accuracy: 0.6087\n",
      "Epoch 31/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1875 - accuracy: 0.6153 - val_loss: 2.2244 - val_accuracy: 0.6106\n",
      "Epoch 32/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1851 - accuracy: 0.6142 - val_loss: 2.2356 - val_accuracy: 0.6083\n",
      "Epoch 33/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1830 - accuracy: 0.6143 - val_loss: 2.2339 - val_accuracy: 0.6089\n",
      "Epoch 34/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1652 - accuracy: 0.6168 - val_loss: 2.2206 - val_accuracy: 0.6067\n",
      "Epoch 35/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1703 - accuracy: 0.6150 - val_loss: 2.2087 - val_accuracy: 0.6051\n",
      "Epoch 36/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1673 - accuracy: 0.6159 - val_loss: 2.1987 - val_accuracy: 0.6048\n",
      "Epoch 37/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1570 - accuracy: 0.6145 - val_loss: 2.2143 - val_accuracy: 0.6061\n",
      "Epoch 38/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1502 - accuracy: 0.6142 - val_loss: 2.2173 - val_accuracy: 0.6055\n",
      "Epoch 41/100\n",
      "3839/3839 [==============================] - 45s 12ms/step - loss: 2.1685 - accuracy: 0.6151 - val_loss: 2.2120 - val_accuracy: 0.6080\n",
      "CPU times: user 12h 52min 24s, sys: 1h 52min 42s, total: 14h 45min 7s\n",
      "Wall time: 30min 50s\n"
     ]
    }
   ],
   "source": [
    "model2 = keras.models.Sequential([\n",
    "    keras.layers.Dense(50, input_shape = (56705,)  , activation = 'relu', kernel_regularizer = regularizers.l2(0.01)),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(100, activation = \"relu\"),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(150, activation = \"relu\"),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),   \n",
    "    keras.layers.Dense(6, activation = \"softmax\")\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model2.compile(loss = \"categorical_crossentropy\", optimizer = \"adam\", metrics = ['accuracy'])\n",
    "\n",
    "# Define the early stopping callback\n",
    "early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)\n",
    "\n",
    "# Train the model with early stopping\n",
    "%time history = model2.fit(X_train_tensor, y_train_enc, batch_size = 32, epochs = 100, validation_data = (X_val_tensor, y_val_enc), callbacks = [early_stop])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "d413626b-0968-48d4-b8bd-93a57b3cd778",
   "metadata": {},
   "source": [
    "### Training evaluation"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "d6394275-efc5-42fa-a078-69930163c2e3",
   "metadata": {},
   "source": [
    "**Loss curve**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "037e5d12-50a0-4549-87f7-2fef6e71f18b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T13:58:35.751866Z",
     "iopub.status.busy": "2023-05-17T13:58:35.751686Z",
     "iopub.status.idle": "2023-05-17T13:58:39.129513Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "960/960 - 3s - loss: 2.2120 - accuracy: 0.6080 - 3s/epoch - 3ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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v6usU6dOeHqe/R/X6OhoMjJq1t8yKSmJ2NhYV7AB6NixI6GhoSQlJQEwdepU7rvvPm666Saef/55Dh486Fr34Ycf5s9//jPXXnstTz31FLt21f2Tu2q5sbjmoX58OCGRGR/vYdG2n3huWRK7usXwwi/n4PPJfbBlLgQ0g/7T3V1VEWmMvPzNVhR3HLca7r33XiZNmsRrr73Gm2++ScuWLRkwYAAAL774Ii+//DKzZs2iS5cuBAQEMGXKFAoLCy9p30Yl/4NoO++W2caNG7njjjt45plnGDRoECEhISxYsIAXX3yxWudhGEaFfVd2zLJbQucuczqd1TrWxY55bvnTTz/Nr371K5YuXcpnn33GU089xYIFC7j11lu57777GDRoEEuXLmX58uXMnDmTF198kcmTJ9eoPpdCLTc/A75enrzw/7ry7PBO2D1s/G/nUYavjuDk9X82V1jzPKx8Ri04IlJ9Npt5e6i+p0vob3Ou22+/HU9PT95//33efvttfv3rX7t+mNetW8fw4cO566676NatG61bt+bAgQOXvO+OHTuSmprK0aNnQ96GDRvKrfPNN9/QsmVLZsyYQUJCAm3btq3wBJe3tzclJSUXPdaOHTvIzc0tt28PDw/atWt3yXWujrLzO3z4sKts7969ZGZmcuWVV7rK2rVrxyOPPMLy5csZOXIkb7559vU/sbGxTJgwgcWLF/Poo4/y73//u07qWkbh5mei7MWb799/DU0DzfFw+q9py8FupfeZv34JPn0EnFX/wxIRaYwCAwMZPXo0TzzxBEePHmX8+PGuZW3atGHFihWsX7+epKQkfvOb35Cenn7J+77pppto3749Y8eOZefOnaxbt44ZM2aUW6dNmzakpqayYMECDh48yCuvvMLHH39cbp1WrVqRnJzMjh07OHHiBA6Ho8KxxowZg6+vL+PGjWPPnj2sWrWKyZMnc/fdd7v629RUSUkJO3bsKDft3buXm266ia5duzJmzBi2bdvGpk2bGDt2LNdffz0JCQnk5+czadIkVq9eTUpKCt988w2bN292BZ8pU6bwxRdfkJyczLZt2/jqq6/KhaK6oHDzM9MrvgmfTr6OHnGhZBUUc9Omq/iqzRMY2GDrm7DoXii+tKZYEZHG5N577+X06dPcdNNNxMXFucqffPJJrrrqKgYNGsQNN9xAVFQUI0aMuOT9enh4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Q0pKyiXvZ8iQIQwZMqTax4+IiCA0NLTa24mIiIj11ajlpk2bNnzyySccPnyYL774goEDBwKQkZFRL510e/ToQXR0NAMGDHCNlHwhDoeDrKyscpOIiIhYV43CzR//+EemTZtGq1at6NWrF4mJiYDZitOjR49areC5oqOjef3111m0aBGLFy+mffv2DBgwgLVr115wm5kzZxISEuKaYmNj66x+IiIi4n41fhQ8PT2dtLQ0unXrhoeHmZE2bdpEcHAwHTp0qH5FbDY+/vhjRowYUa3thg0bhs1mY8mSJZUudzgcOBwO13xWVhaxsbF6FFxERKQRqc6j4DXqcwMQFRVFVFQUP/30EzabjebNm7tlAL9rrrmG+fPnX3C5j4+P61URIiIiYn01ui3ldDp59tlnCQkJoWXLlsTFxREaGsqf/vQnnE5nbdexStu3byc6OrpejykiIiINV41abmbMmMHcuXN5/vnnufbaazEMg2+++Yann36agoICnnvuuUvaT05ODj/88INrPjk5mR07dtCkSRPi4uKYPn06R44ccY2pM2vWLFq1akWnTp0oLCxk/vz5LFq0iEWLFtXkNERERMSCahRu3n77bd544w3X28ABunXrRvPmzXnooYcuOdxs2bKF/v37u+anTp0KwLhx43jrrbdIS0sjNTXVtbywsJBp06Zx5MgR/Pz86NSpE0uXLmXo0KE1OQ0RERGxoBp1KPb19WXXrl20a9euXPn+/fvp3r07+fn5tVbB2qZ3S4mIiDQ+df5uqQuNKvzqq6/StWvXmuxSREREpFbU6LbU3/72N2655RZWrlxJYmIiNpuN9evXc/jwYZYtW1bbdRQRERG5ZDVqubn++uv5/vvvufXWWzlz5gynTp1i5MiRfPfdd7z55pu1XUcRERGRS1bjQfwqs3PnTq666ipKSkpqa5e1Tn1uREREGp8673MjIiIi0lAp3IiIiIilKNyIiIiIpVTraamRI0dWufzMmTOXUxcRERGRy1atcBMSEnLR5WPHjr2sComIiIhcjmqFGz3mLSIiIg2d+tyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpSjciIiIiKUo3IiIiIilKNyIiIiIpbg13Kxdu5Zhw4YRExODzWbjk08+ueg2a9asoWfPnvj6+tK6dWvmzJlT9xUVERGRRsOt4SY3N5du3brx6quvXtL6ycnJDB06lL59+7J9+3aeeOIJHn74YRYtWlTHNRUREZHGwu7Ogw8ZMoQhQ4Zc8vpz5swhLi6OWbNmAXDllVeyZcsWXnjhBUaNGlVHtRQREZHGpFH1udmwYQMDBw4sVzZo0CC2bNlCUVGRm2olIiIiDYlbW26qKz09ncjIyHJlkZGRFBcXc+LECaKjoyts43A4cDgcrvmsrKw6r6eIiIi4T6NquQGw2Wzl5g3DqLS8zMyZMwkJCXFNsbGxdV5HERERcZ9GFW6ioqJIT08vV5aRkYHdbic8PLzSbaZPn05mZqZrOnz4cH1UVURERNykUd2WSkxM5H//+1+5suXLl5OQkICXl1el2/j4+ODj41Mf1RMREZEGwK0tNzk5OezYsYMdO3YA5qPeO3bsIDU1FTBbXcaOHetaf8KECaSkpDB16lSSkpKYN28ec+fOZdq0ae6ovoiIiDRAbm252bJlC/3793fNT506FYBx48bx1ltvkZaW5go6APHx8SxbtoxHHnmE1157jZiYGF555RU9Bi4iIiIuNqOsR+7PRFZWFiEhIWRmZhIcHOzu6oiIiMglqM7vd6PqUCwiIiJyMQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpCjciIiJiKQo3IiIiYikKNyIiImIpbg83//znP4mPj8fX15eePXuybt26C667evVqbDZbhWnfvn31WGMRERFpyNwabhYuXMiUKVOYMWMG27dvp2/fvgwZMoTU1NQqt9u/fz9paWmuqW3btvVUYxEREWno3BpuXnrpJe69917uu+8+rrzySmbNmkVsbCyzZ8+ucruIiAiioqJck6enZz3VWERERBo6t4WbwsJCtm7dysCBA8uVDxw4kPXr11e5bY8ePYiOjmbAgAGsWrWqynUdDgdZWVnlJhEREbEut4WbEydOUFJSQmRkZLnyyMhI0tPTK90mOjqa119/nUWLFrF48WLat2/PgAEDWLt27QWPM3PmTEJCQlxTbGxsrZ6HiIiINCx2d1fAZrOVmzcMo0JZmfbt29O+fXvXfGJiIocPH+aFF16gX79+lW4zffp0pk6d6prPyspSwBEREbEwt7XcNG3aFE9PzwqtNBkZGRVac6pyzTXXcODAgQsu9/HxITg4uNwkIiIi1uW2cOPt7U3Pnj1ZsWJFufIVK1bQp0+fS97P9u3biY6Oru3qiYiISCPl1ttSU6dO5e677yYhIYHExERef/11UlNTmTBhAmDeUjpy5AjvvPMOALNmzaJVq1Z06tSJwsJC5s+fz6JFi1i0aJE7T0NEREQaELeGm9GjR3Py5EmeffZZ0tLS6Ny5M8uWLaNly5YApKWllRvzprCwkGnTpnHkyBH8/Pzo1KkTS5cuZejQoe46BREREWlgbIZhGO6uRH3KysoiJCSEzMxM9b8RERFpJKrz++321y+IiIiI1CaFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFIUbERERsRSFGxEREbEUhRsRERGxFLeHm3/+85/Ex8fj6+tLz549WbduXZXrr1mzhp49e+Lr60vr1q2ZM2dOPdVUREREGgO3hpuFCxcyZcoUZsyYwfbt2+nbty9DhgwhNTW10vWTk5MZOnQoffv2Zfv27TzxxBM8/PDDLFq0qJ5rLiIiIg2VzTAMw10H7927N1dddRWzZ892lV155ZWMGDGCmTNnVlj/scceY8mSJSQlJbnKJkyYwM6dO9mwYcMlHTMrK4uQkBAyMzMJDg6+/JMQERGROled3297PdWpgsLCQrZu3crjjz9ernzgwIGsX7++0m02bNjAwIEDy5UNGjSIuXPnUlRUhJeXV4VtHA4HDofDNZ+ZmQmYf0giIiLSOJT9bl9Km4zbws2JEycoKSkhMjKyXHlkZCTp6emVbpOenl7p+sXFxZw4cYLo6OgK28ycOZNnnnmmQnlsbOxl1F5ERETcITs7m5CQkCrXcVu4KWOz2crNG4ZRoexi61dWXmb69OlMnTrVNe90Ojl16hTh4eFVHqcmsrKyiI2N5fDhw5a85WX18wPrn6POr/Gz+jnq/Bq/ujpHwzDIzs4mJibmouu6Ldw0bdoUT0/PCq00GRkZFVpnykRFRVW6vt1uJzw8vNJtfHx88PHxKVcWGhpa84pfguDgYMv+pQXrnx9Y/xx1fo2f1c9R59f41cU5XqzFpozbnpby9vamZ8+erFixolz5ihUr6NOnT6XbJCYmVlh/+fLlJCQkVNrfRkRERH5+3Poo+NSpU3njjTeYN28eSUlJPPLII6SmpjJhwgTAvKU0duxY1/oTJkwgJSWFqVOnkpSUxLx585g7dy7Tpk1z1ymIiIhIA+PWPjejR4/m5MmTPPvss6SlpdG5c2eWLVtGy5YtAUhLSys35k18fDzLli3jkUce4bXXXiMmJoZXXnmFUaNGuesUyvHx8eGpp56qcBvMKqx+fmD9c9T5NX5WP0edX+PXEM7RrePciIiIiNQ2t79+QURERKQ2KdyIiIiIpSjciIiIiKUo3IiIiIilKNzUkn/+85/Ex8fj6+tLz549WbdunburVGuefvppbDZbuSkqKsrd1aqxtWvXMmzYMGJiYrDZbHzyySfllhuGwdNPP01MTAx+fn7ccMMNfPfdd+6pbA1d7BzHjx9f4Zpec8017qlsDcycOZOrr76aoKAgIiIiGDFiBPv37y+3TmO+jpdyfo35Gs6ePZuuXbu6BnlLTEzks88+cy1vzNeuzMXOsTFfv8rMnDkTm83GlClTXGXuvI4KN7Vg4cKFTJkyhRkzZrB9+3b69u3LkCFDyj3G3th16tSJtLQ017R79253V6nGcnNz6datG6+++mqly//2t7/x0ksv8eqrr7J582aioqK4+eabyc7Oruea1tzFzhFg8ODB5a7psmXL6rGGl2fNmjVMnDiRjRs3smLFCoqLixk4cCC5ubmudRrzdbyU84PGew1btGjB888/z5YtW9iyZQs33ngjw4cPd/3wNeZrV+Zi5wiN9/qdb/Pmzbz++ut07dq1XLlbr6Mhl61Xr17GhAkTypV16NDBePzxx91Uo9r11FNPGd26dXN3NeoEYHz88ceueafTaURFRRnPP/+8q6ygoMAICQkx5syZ44YaXr7zz9EwDGPcuHHG8OHD3VKfupCRkWEAxpo1awzDsN51PP/8DMN61zAsLMx44403LHftzlV2joZhneuXnZ1ttG3b1lixYoVx/fXXG7/97W8Nw3D/v0G13FymwsJCtm7dysCBA8uVDxw4kPXr17upVrXvwIEDxMTEEB8fzx133MGPP/7o7irVieTkZNLT08tdTx8fH66//npLXU+A1atXExERQbt27bj//vvJyMhwd5VqLDMzE4AmTZoA1ruO559fGStcw5KSEhYsWEBubi6JiYmWu3ZQ8RzLWOH6TZw4kVtuuYWbbrqpXLm7r6Pb3wre2J04cYKSkpIKL/uMjIys8JLPxqp379688847tGvXjmPHjvHnP/+ZPn368N13313whaWNVdk1q+x6pqSkuKNKdWLIkCH8v//3/2jZsiXJyck8+eST3HjjjWzdurXRjZxqGAZTp07luuuuo3PnzoC1rmNl5weN/xru3r2bxMRECgoKCAwM5OOPP6Zjx46uHz4rXLsLnSM0/usHsGDBArZt28bmzZsrLHP3v0GFm1pis9nKzRuGUaGssRoyZIjre5cuXUhMTOSKK67g7bffZurUqW6sWd2x8vUE89UnZTp37kxCQgItW7Zk6dKljBw50o01q75Jkyaxa9cuvv766wrLrHAdL3R+jf0atm/fnh07dnDmzBkWLVrEuHHjWLNmjWu5Fa7dhc6xY8eOjf76HT58mN/+9rcsX74cX1/fC67nruuo21KXqWnTpnh6elZopcnIyKiQWK0iICCALl26cODAAXdXpdaVPQX2c7qeANHR0bRs2bLRXdPJkyezZMkSVq1aRYsWLVzlVrmOFzq/yjS2a+jt7U2bNm1ISEhg5syZdOvWjb///e+WuXZw4XOsTGO7flu3biUjI4OePXtit9ux2+2sWbOGV155Bbvd7rpW7rqOCjeXydvbm549e7JixYpy5StWrKBPnz5uqlXdcjgcJCUlER0d7e6q1Lr4+HiioqLKXc/CwkLWrFlj2esJcPLkSQ4fPtxorqlhGEyaNInFixfz1VdfER8fX255Y7+OFzu/yjS2a3g+wzBwOByN/tpVpewcK9PYrt+AAQPYvXs3O3bscE0JCQmMGTOGHTt20Lp1a/dexzrvsvwzsGDBAsPLy8uYO3eusXfvXmPKlClGQECAcejQIXdXrVY8+uijxurVq40ff/zR2Lhxo/GLX/zCCAoKarTnl52dbWzfvt3Yvn27ARgvvfSSsX37diMlJcUwDMN4/vnnjZCQEGPx4sXG7t27jTvvvNOIjo42srKy3FzzS1fVOWZnZxuPPvqosX79eiM5OdlYtWqVkZiYaDRv3rzRnOODDz5ohISEGKtXrzbS0tJcU15enmudxnwdL3Z+jf0aTp8+3Vi7dq2RnJxs7Nq1y3jiiScMDw8PY/ny5YZhNO5rV6aqc2zs1+9Czn1ayjDcex0VbmrJa6+9ZrRs2dLw9vY2rrrqqnKPbDZ2o0ePNqKjow0vLy8jJibGGDlypPHdd9+5u1o1tmrVKgOoMI0bN84wDPMRxqeeesqIiooyfHx8jH79+hm7d+92b6WrqapzzMvLMwYOHGg0a9bM8PLyMuLi4oxx48YZqamp7q72Javs3ADjzTffdK3TmK/jxc6vsV/De+65x/Xfy2bNmhkDBgxwBRvDaNzXrkxV59jYr9+FnB9u3HkdbYZhGHXfPiQiIiJSP9TnRkRERCxF4UZEREQsReFGRERELEXhRkRERCxF4UZEREQsReFGRERELEXhRkRERCxF4UZEBPMFf5988om7qyEitUDhRkTcbvz48dhstgrT4MGD3V01EWmE7O6ugIgIwODBg3nzzTfLlfn4+LipNiLSmKnlRkQaBB8fH6KiospNYWFhgHnLaPbs2QwZMgQ/Pz/i4+P58MMPy22/e/dubrzxRvz8/AgPD+eBBx4gJyen3Drz5s2jU6dO+Pj4EB0dzaRJk8otP3HiBLfeeiv+/v60bduWJUuW1O1Ji0idULgRkUbhySefZNSoUezcuZO77rqLO++8k6SkJADy8vIYPHgwYWFhbN68mQ8//JCVK1eWCy+zZ89m4sSJPPDAA+zevZslS5bQpk2bcsd45plnuP3229m1axdDhw5lzJgxnDp1ql7PU0RqQb28nlNEpArjxo0zPD09jYCAgHLTs88+axiG+ZbsCRMmlNumd+/exoMPPmgYhmG8/vrrRlhYmJGTk+NavnTpUsPDw8NIT083DMMwYmJijBkzZlywDoDxhz/8wTWfk5Nj2Gw247PPPqu18xSR+qE+NyLSIPTv35/Zs2eXK2vSpInre2JiYrlliYmJ7NixA4CkpCS6detGQECAa/m1116L0+lk//792Gw2jh49yoABA6qsQ9euXV3fAwICCAoKIiMjo6anJCJuonAjIg1CQEBAhdtEF2Oz2QAwDMP1vbJ1/Pz8Lml/Xl5eFbZ1Op3VqpOIuJ/63IhIo7Bx48YK8x06dACgY8eO7Nixg9zcXNfyb775Bg8PD9q1a0dQUBCtWrXiyy+/rNc6i4h7qOVGRBoEh8NBenp6uTK73U7Tpk0B+PDDD0lISOC6667jvffeY9OmTcydOxeAMWPG8NRTTzFu3Diefvppjh8/zuTJk7n77ruJjIwE4Omnn2bChAlEREQwZMgQsrOz+eabb5g8eXL9nqiI1DmFGxFpED7//HOio6PLlbVv3559+/YB5pNMCxYs4KGHHiIqKor33nuPjh07AuDv788XX3zBb3/7W66++mr8/f0ZNWoUL730kmtf48aNo6CggJdffplp06bRtGlTbrvttvo7QRGpNzbDMAx3V0JEpCo2m42PP/6YESNGuLsqItIIqM+NiIiIWIrCjYiIiFiK+tyISIOnu+ciUh1quRERERFLUbgRERERS1G4EREREUtRuBERERFLUbgRERERS1G4EREREUtRuBERERFLUbgRERERS1G4EREREUv5//8vvfX3toeAAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Evaluate loss (train vs. val)\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label = 'Validation Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.ylim([0, 3])\n",
    "plt.legend(loc='upper right')\n",
    "\n",
    "validation_loss, validation_acc = model2.evaluate(X_val_tensor,  y_val_enc, verbose=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "9282de79-1e85-4db9-a619-47e687301df5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-18T12:04:18.151151Z",
     "iopub.status.busy": "2023-05-18T12:04:18.150374Z",
     "iopub.status.idle": "2023-05-18T12:04:18.682659Z",
     "shell.execute_reply": "2023-05-18T12:04:18.681779Z",
     "shell.execute_reply.started": "2023-05-18T12:04:18.151094Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Save model 2\n",
    "pickle.dump(model2, open('MLP_TFIDF.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "11381bd9-dc58-4b16-870e-56bf812f5144",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-18T12:05:03.703109Z",
     "iopub.status.busy": "2023-05-18T12:05:03.702350Z",
     "iopub.status.idle": "2023-05-18T12:05:04.418849Z",
     "shell.execute_reply": "2023-05-18T12:05:04.417983Z",
     "shell.execute_reply.started": "2023-05-18T12:05:03.703054Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load model 2\n",
    "model2 = pd.read_pickle(open('MLP_TFIDF.pkl', 'rb'))"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "2c922a49-9ec2-4088-8755-cf7df9171dac",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on the test set (out-of-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "2b282ebf-529f-495a-8f46-8659acb6103f",
   "metadata": {},
   "source": [
    "**Predict on the test set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "85200064-b141-4993-ab4d-ef8f9f2175df",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T14:43:06.568811Z",
     "iopub.status.busy": "2023-05-17T14:43:06.568017Z",
     "iopub.status.idle": "2023-05-17T14:43:58.303428Z",
     "shell.execute_reply": "2023-05-17T14:43:58.301825Z",
     "shell.execute_reply.started": "2023-05-17T14:43:06.568751Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Convert X test csr-matrix into tensor \n",
    "X_test_tensor = tf.convert_to_tensor(X_test_scaled.toarray())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "28caf66d-7951-4c3d-bd28-e04bbd90c92c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T14:44:37.983824Z",
     "iopub.status.busy": "2023-05-17T14:44:37.983065Z",
     "iopub.status.idle": "2023-05-17T14:44:51.666580Z",
     "shell.execute_reply": "2023-05-17T14:44:51.665620Z",
     "shell.execute_reply.started": "2023-05-17T14:44:37.983766Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3571/3571 [==============================] - 13s 4ms/step\n"
     ]
    }
   ],
   "source": [
    "# Predict the labels for the test set (enc)\n",
    "y_pred_prob_enc = model2.predict(X_test_tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7425e59b-2319-4efe-9b52-e62569ce7732",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T14:44:52.164834Z",
     "iopub.status.busy": "2023-05-17T14:44:52.164378Z",
     "iopub.status.idle": "2023-05-17T14:44:52.173672Z",
     "shell.execute_reply": "2023-05-17T14:44:52.172810Z",
     "shell.execute_reply.started": "2023-05-17T14:44:52.164801Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Convert probabilities to class labels\n",
    "y_pred = np.argmax(y_pred_prob_enc, axis=1)\n",
    "y_test_labels = np.argmax(y_test_enc, axis=1)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "0832deb5-95cb-495a-b76d-add817a1ae9e",
   "metadata": {},
   "source": [
    "**Confusion Matrix**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "83947a1d-732f-4894-bbe0-429d5087831c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T14:44:53.507739Z",
     "iopub.status.busy": "2023-05-17T14:44:53.506946Z",
     "iopub.status.idle": "2023-05-17T14:44:53.538470Z",
     "shell.execute_reply": "2023-05-17T14:44:53.537554Z",
     "shell.execute_reply.started": "2023-05-17T14:44:53.507682Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+---------+---------+---------+---------+---------+---------+---------+\n",
      "|         | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |\n",
      "+---------+---------+---------+---------+---------+---------+---------+\n",
      "| Class 0 |  29977  |   2523  |   2021  |   4908  |   1119  |   480   |\n",
      "| Class 1 |   3475  |  22570  |   1244  |   3428  |   903   |   494   |\n",
      "| Class 2 |   1066  |   587   |   4127  |   812   |   221   |   305   |\n",
      "| Class 3 |   4288  |   2104  |   1127  |  15511  |   882   |   286   |\n",
      "| Class 4 |   1994  |   1198  |   607   |   1835  |   1390  |   463   |\n",
      "| Class 5 |   255   |   252   |   372   |   254   |   150   |   1043  |\n",
      "+---------+---------+---------+---------+---------+---------+---------+\n"
     ]
    }
   ],
   "source": [
    "# Compute the confusion matrix\n",
    "confusion_mtx = confusion_matrix(y_test_labels, y_pred)\n",
    "\n",
    "# Create PrettyTable table object\n",
    "table = PrettyTable()\n",
    "table.field_names = [\"\", \"Class 0\", \"Class 1\", \"Class 2\", \"Class 3\", \"Class 4\", \"Class 5\"]\n",
    "\n",
    "# Add the rows to the table\n",
    "for i in range(len(confusion_mtx)):\n",
    "    row = [\"Class \" + str(i)]\n",
    "    row.extend(confusion_mtx[i])\n",
    "    table.add_row(row)\n",
    "\n",
    "# Print the confusion matrix table \n",
    "print(table)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "76aa2670-d916-411e-8522-24967a597694",
   "metadata": {},
   "source": [
    "**Classification report**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e7b5e258-e67d-493d-b6ff-347b531aa7a6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T14:44:54.650862Z",
     "iopub.status.busy": "2023-05-17T14:44:54.650129Z",
     "iopub.status.idle": "2023-05-17T14:44:54.832605Z",
     "shell.execute_reply": "2023-05-17T14:44:54.831727Z",
     "shell.execute_reply.started": "2023-05-17T14:44:54.650803Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+--------------+-----------+--------+----------+---------+\n",
      "|    Class     | Precision | Recall | F1-score | Support |\n",
      "+--------------+-----------+--------+----------+---------+\n",
      "|      0       |    0.73   |  0.73  |   0.73   |  41028  |\n",
      "|      1       |    0.77   |  0.7   |   0.74   |  32114  |\n",
      "|      2       |    0.43   |  0.58  |   0.5    |   7118  |\n",
      "|      3       |    0.58   |  0.64  |   0.61   |  24198  |\n",
      "|      4       |    0.3    |  0.19  |   0.23   |   7487  |\n",
      "|      5       |    0.34   |  0.45  |   0.39   |   2326  |\n",
      "|  macro avg   |    0.53   |  0.55  |   0.53   |         |\n",
      "| weighted avg |    0.66   |  0.65  |   0.65   |         |\n",
      "+--------------+-----------+--------+----------+---------+\n"
     ]
    }
   ],
   "source": [
    "# Generate the classification report\n",
    "class_report = classification_report(y_test_labels, y_pred, output_dict=True)\n",
    "\n",
    "# Create the PrettyTable\n",
    "table = PrettyTable()\n",
    "table.field_names = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']\n",
    "\n",
    "# Loop through each class in the classification report and add its metrics to the table\n",
    "for class_name, metrics in class_report.items():\n",
    "    if class_name.isnumeric():\n",
    "        class_id = int(class_name)\n",
    "        precision = round(metrics['precision'], 2)\n",
    "        recall = round(metrics['recall'], 2)\n",
    "        f1_score = round(metrics['f1-score'], 2)\n",
    "        support = metrics['support']\n",
    "        table.add_row([class_id, precision, recall, f1_score, support])\n",
    "\n",
    "# Add the macro and weighted averages to the table\n",
    "macro_precision = round(class_report['macro avg']['precision'], 2)\n",
    "macro_recall = round(class_report['macro avg']['recall'], 2)\n",
    "macro_f1_score = round(class_report['macro avg']['f1-score'], 2)\n",
    "table.add_row(['macro avg', macro_precision, macro_recall, macro_f1_score, ''])\n",
    "\n",
    "weighted_precision = round(class_report['weighted avg']['precision'], 2)\n",
    "weighted_recall = round(class_report['weighted avg']['recall'], 2)\n",
    "weighted_f1_score = round(class_report['weighted avg']['f1-score'], 2)\n",
    "table.add_row(['weighted avg', weighted_precision, weighted_recall, weighted_f1_score, ''])\n",
    "\n",
    "# Print the table\n",
    "print(table)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
