{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "63c73c67-1cb0-41bf-a944-0026c00957ec",
   "metadata": {},
   "source": [
    "### Install dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "00e9661e",
   "metadata": {},
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
    "pip install tensorflow scikit-learn pandas matplotlib prettytable gensim --quiet"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "84de7e84-8dd4-43a9-84f1-86bd0a6d63d2",
   "metadata": {},
   "source": [
    "### Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89bc0980",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:39:15.624977Z",
     "iopub.status.busy": "2023-05-16T19:39:15.624107Z",
     "iopub.status.idle": "2023-05-16T19:39:15.636939Z",
     "shell.execute_reply": "2023-05-16T19:39:15.634722Z",
     "shell.execute_reply.started": "2023-05-16T19:39:15.624937Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import gensim\n",
    "import time\n",
    "import tensorflow as tf\n",
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
    "from tensorflow.keras.layers import BatchNormalization\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout\n",
    "from tensorflow.keras.utils import to_categorical\n"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "ab55bd5e-100d-4307-9009-c9f11a928ba9",
   "metadata": {},
   "source": [
    "### Load and split dataset with Word2Vec embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aebad4e9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:38:33.820215Z",
     "iopub.status.busy": "2023-05-16T19:38:33.819471Z",
     "iopub.status.idle": "2023-05-16T19:38:49.282820Z",
     "shell.execute_reply": "2023-05-16T19:38:49.281250Z",
     "shell.execute_reply.started": "2023-05-16T19:38:33.820169Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load dense\n",
    "X_train_dense, X_test_dense, y_train_dense, y_test_dense, w2v_train, w2v_test = pd.read_pickle(\"pickles/dense.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7ce84a15-1eab-49ab-8bc4-9bac4a5b9389",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:38:49.309082Z",
     "iopub.status.busy": "2023-05-16T19:38:49.308623Z",
     "iopub.status.idle": "2023-05-16T19:38:49.469166Z",
     "shell.execute_reply": "2023-05-16T19:38:49.466941Z",
     "shell.execute_reply.started": "2023-05-16T19:38:49.309046Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Split data into train and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train_dense, y_train_dense, test_size=0.8, random_state=42)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "4bf01906-185d-4793-bae0-b5a40584cb55",
   "metadata": {},
   "source": [
    "### Scaling and encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "42a9693c-c10d-44f6-9bf4-8f0763fba6fa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:39:00.903767Z",
     "iopub.status.busy": "2023-05-16T19:39:00.903356Z",
     "iopub.status.idle": "2023-05-16T19:39:02.359384Z",
     "shell.execute_reply": "2023-05-16T19:39:02.354481Z",
     "shell.execute_reply.started": "2023-05-16T19:39:00.903740Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# One-hot encode labels \n",
    "encoder = LabelBinarizer()\n",
    "y_train_enc = encoder.fit_transform(y_train)\n",
    "y_val_enc = encoder.fit_transform(y_val)\n",
    "y_test_enc = encoder.fit_transform(y_test_dense)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "15701825-1d37-4866-acc0-e95bcf1b27de",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:39:02.391210Z",
     "iopub.status.busy": "2023-05-16T19:39:02.390911Z",
     "iopub.status.idle": "2023-05-16T19:39:04.373475Z",
     "shell.execute_reply": "2023-05-16T19:39:04.372328Z",
     "shell.execute_reply.started": "2023-05-16T19:39:02.391188Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Scale the w2w feature vectors\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_val_scaled = scaler.fit_transform(X_val)\n",
    "X_test_scaled = scaler.fit_transform(X_test_dense)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "99f66b12-676c-4e51-b15f-1a3d061d741e",
   "metadata": {},
   "source": [
    "### Build and train the Word2Vec MLP classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1dbcc22e-57ed-4928-8c0a-8c946f493101",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:51:18.845120Z",
     "iopub.status.busy": "2023-05-16T19:51:18.844274Z",
     "iopub.status.idle": "2023-05-16T19:54:37.923860Z",
     "shell.execute_reply": "2023-05-16T19:54:37.921593Z",
     "shell.execute_reply.started": "2023-05-16T19:51:18.845067Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "960/960 [==============================] - 17s 15ms/step - loss: 1.3188 - accuracy: 0.5350 - val_loss: 1.0648 - val_accuracy: 0.6214\n",
      "Epoch 2/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 1.0976 - accuracy: 0.6077 - val_loss: 1.0266 - val_accuracy: 0.6352\n",
      "Epoch 3/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 1.0414 - accuracy: 0.6277 - val_loss: 1.0267 - val_accuracy: 0.6366\n",
      "Epoch 4/100\n",
      "960/960 [==============================] - 14s 14ms/step - loss: 1.0183 - accuracy: 0.6362 - val_loss: 1.0059 - val_accuracy: 0.6429\n",
      "Epoch 5/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 1.0044 - accuracy: 0.6418 - val_loss: 1.0106 - val_accuracy: 0.6437\n",
      "Epoch 6/100\n",
      "960/960 [==============================] - 14s 14ms/step - loss: 0.9861 - accuracy: 0.6468 - val_loss: 0.9983 - val_accuracy: 0.6466\n",
      "Epoch 7/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 0.9752 - accuracy: 0.6504 - val_loss: 1.0041 - val_accuracy: 0.6444\n",
      "Epoch 8/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 0.9645 - accuracy: 0.6544 - val_loss: 1.0018 - val_accuracy: 0.6472\n",
      "Epoch 9/100\n",
      "960/960 [==============================] - 14s 14ms/step - loss: 0.9511 - accuracy: 0.6615 - val_loss: 0.9958 - val_accuracy: 0.6482\n",
      "Epoch 10/100\n",
      "960/960 [==============================] - 14s 14ms/step - loss: 0.9388 - accuracy: 0.6599 - val_loss: 1.0097 - val_accuracy: 0.6469\n",
      "Epoch 11/100\n",
      "960/960 [==============================] - 14s 14ms/step - loss: 0.9278 - accuracy: 0.6676 - val_loss: 0.9984 - val_accuracy: 0.6475\n",
      "Epoch 12/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 0.9178 - accuracy: 0.6706 - val_loss: 1.0056 - val_accuracy: 0.6485\n",
      "Epoch 13/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 0.9126 - accuracy: 0.6726 - val_loss: 1.0135 - val_accuracy: 0.6489\n",
      "Epoch 14/100\n",
      "960/960 [==============================] - 14s 15ms/step - loss: 0.9022 - accuracy: 0.6741 - val_loss: 1.0086 - val_accuracy: 0.6468\n",
      "CPU times: user 4min 19s, sys: 44.4 s, total: 5min 4s\n",
      "Wall time: 3min 18s\n"
     ]
    }
   ],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Dense(64, input_shape=(300,), activation='tanh'),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(128, activation='tanh'),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(256, activation='tanh'),\n",
    "    keras.layers.BatchNormalization(),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(6, activation='softmax')\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model.compile(loss = \"categorical_crossentropy\", optimizer = \"adam\", metrics = ['accuracy'])\n",
    "\n",
    "# Define the early stopping callback\n",
    "early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)\n",
    "\n",
    "# Train the model with early stopping\n",
    "%time history = model.fit(X_train_scaled, y_train_enc, batch_size = 32, epochs = 100, validation_data = (X_val_scaled, y_val_enc), callbacks = [early_stop])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "44546be9-331c-4111-80de-df2e71ee5960",
   "metadata": {},
   "source": [
    "### Training evaluation"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "b20727bf-d4e4-4404-ae71-b0c6ed93f62e",
   "metadata": {},
   "source": [
    "**Loss**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e99054e6-a602-4e6f-aded-f3ae9f346b49",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:50:24.658382Z",
     "iopub.status.busy": "2023-05-16T19:50:24.657705Z",
     "iopub.status.idle": "2023-05-16T19:50:33.079283Z",
     "shell.execute_reply": "2023-05-16T19:50:33.077182Z",
     "shell.execute_reply.started": "2023-05-16T19:50:24.658323Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3838/3838 - 8s - loss: 0.9673 - accuracy: 0.6570 - 8s/epoch - 2ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "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2hctWqQJEyZIkrKzs5WVleVbVlJSomnTpunw4cMKDQ1Vjx499N5772nkyJFNVTYAAGjGms2A4qZSlwFJAACgeWhxA4oBAAAaCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYCuEGAABYSkDDzezZs9WvXz9FREQoNjZWo0aN0v79+8+63rp169S3b1+FhIQoJSVFL730UhNUCwAAWoKAhpt169Zp0qRJ2rx5s1atWqWysjINHTpUhYWFZ1wnMzNTI0eO1KBBg7Rz507NnDlT999/vzIyMpqwcgAA0FwZpmmagS6iwg8//KDY2FitW7dOV111VbV9pk+fruXLl2vfvn2+trS0NO3evVubNm0662fk5+crKipKeXl5ioyMbLDaAQBA46nL93ezGnOTl5cnSYqOjj5jn02bNmno0KF+bcOGDdO2bdtUWlpapb/L5VJ+fr7fBAAArKvZhBvTNJWenq4rr7xSF1988Rn75eTkKC4uzq8tLi5OZWVlOnLkSJX+s2fPVlRUlG9KTExs8NoBAEDz0WzCzeTJk7Vnzx698cYbZ+1rGIbf+4oza6e3S9KMGTOUl5fnmw4dOtQwBQMAgGbJEegCJGnKlClavny51q9frw4dOtTYNz4+Xjk5OX5tubm5cjgciomJqdLf6XTK6XQ2aL0AAKD5CuiRG9M0NXnyZC1btkwfffSRkpOTz7rOgAEDtGrVKr+2lStXKjU1VUFBQY1VKgAAaCECGm4mTZqkJUuW6PXXX1dERIRycnKUk5OjkydP+vrMmDFD48aN871PS0vTwYMHlZ6ern379umVV17RwoULNW3atEDsAgAAaGYCGm7mz5+vvLw8XXPNNUpISPBNS5cu9fXJzs5WVlaW731ycrJWrFihtWvXqnfv3vrtb3+ruXPnavTo0YHYBQAA0Mw0q/vcNAXucwMAQMvTYu9zAwAAcK4INwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFIINwAAwFLqFW4OHTqkb7/91vd+y5Ytmjp1qhYsWNBghQEAANRHvcLNz3/+c61Zs0aSlJOTo+uuu05btmzRzJkz9cQTTzRogQAAAHVRr3Dz6aefqn///pKkv//977r44ou1ceNGvf7661q8eHFD1gcAAFAn9Qo3paWlcjqdkqTVq1frv//7vyVJ3bp1U3Z2dsNVBwAAUEf1Cjc9evTQSy+9pA0bNmjVqlUaPny4JOm7775TTExMgxYIAABQF/UKN88884z+/Oc/65prrtFtt92mXr16SZKWL1/uO10FAAAQCIZpmmZ9VnS73crPz1ebNm18bd98843CwsIUGxvbYAU2tPz8fEVFRSkvL0+RkZGBLgcAANRCXb6/63Xk5uTJk3K5XL5gc/DgQc2ZM0f79+9v1sEGAABYX73CzY033qhXX31VknT8+HFddtlleu655zRq1CjNnz+/1ttZv369brjhBrVv316GYejdd9+tsf/atWtlGEaV6YsvvqjPbgAAAAuqV7jZsWOHBg0aJEl6++23FRcXp4MHD+rVV1/V3Llza72dwsJC9erVS/PmzavT5+/fv1/Z2dm+qUuXLnVaHwAAWJejPisVFRUpIiJCkrRy5UrddNNNstlsuvzyy3Xw4MFab2fEiBEaMWJEnT8/NjZWrVu3rvN6AADA+up15KZz58569913dejQIX344YcaOnSoJCk3N7dJBun26dNHCQkJGjx4sO9OyWficrmUn5/vNwEAAOuqV7j59a9/rWnTpqlTp07q37+/BgwYIMl7FKdPnz4NWmBlCQkJWrBggTIyMrRs2TJ17dpVgwcP1vr168+4zuzZsxUVFeWbEhMTG60+AAAQePW+FDwnJ0fZ2dnq1auXbDZvRtqyZYsiIyPVrVu3uhdiGHrnnXc0atSoOq13ww03yDAMLV++vNrlLpdLLpfL9z4/P1+JiYlcCg4AQAtSl0vB6zXmRpLi4+MVHx+vb7/9VoZh6IILLgjIDfwuv/xyLVmy5IzLnU6n71ERAADA+up1Wsrj8eiJJ55QVFSUOnbsqKSkJLVu3Vq//e1v5fF4GrrGGu3cuVMJCQlN+pkAAKD5qteRm1mzZmnhwoV6+umndcUVV8g0TX3yySd67LHHVFxcrCeffLJW2ykoKNCBAwd87zMzM7Vr1y5FR0crKSlJM2bM0OHDh3331JkzZ446deqkHj16qKSkREuWLFFGRoYyMjLqsxsAAMCC6hVu/vrXv+ovf/mL72ngktSrVy9dcMEFuu+++2odbrZt26Zrr73W9z49PV2SNH78eC1evFjZ2dnKysryLS8pKdG0adN0+PBhhYaGqkePHnrvvfc0cuTI+uwGAACwoHoNKA4JCdGePXt04YUX+rXv379fvXv31smTJxuswIbGs6UAAGh5Gv3ZUme6q/C8efPUs2fP+mwSAACgQdTrtNSzzz6r66+/XqtXr9aAAQNkGIY2btyoQ4cOacWKFQ1dIwAAQK3V68jN1Vdfrf/85z/66U9/quPHj+vYsWO66aab9Nlnn2nRokUNXSMAAECt1fsmftXZvXu3Lr30Urnd7obaZINjzA0AAC1Po4+5AQAAaK4INwAAwFIINw3oi5x8HcgtCHQZAACc1+p0tdRNN91U4/Ljx4+fSy0t2t5v8zTmL5vVyunQ2/cOVPvWoYEuCQCA81Kdwk1UVNRZl48bN+6cCmqp2rcOUdsIp77+oVBjF/5bb6UNVHR4cKDLAgDgvNOgV0u1BI15tdR3x0/qZ/M36ru8YvXsEKXX775crZz1fvA6AAAox9VSAdK+dahenXiZ2oQFac+3ebrn1W1ylTXfy+IBALAiwk0D6xzbSovv6K/wYLs2fnVUD7yxS27PeXVwDACAgCLcNIJeia21YFyqgu02ffBZjmYu26vz7OwfAAABQ7hpJFd0bqu5t/WWzZCWbjukZz7YH+iSAAA4LxBuGtHwixM0+6ZLJEkvrftKf173VYArAgDA+gg3jeyWfkl6ZEQ3SdLs97/Q37ceCnBFAABYG+GmCaRd/f/0i6tSJEmPLNujDz/LCXBFAABYF+GmiTwyoptuSU2Ux5SmvL5TG786EuiSAACwJMJNEzEMQ0/+9GIN6xGnErdHd/91m/Z8ezzQZQEAYDmEmybksNv0h1v7aEBKjApL3JqwaKu++oEHbQIA0JAIN00sJMiuBeP66pILonSssERj//JvfXf8ZKDLAgDAMgg3ARAREqTFd/RTSrtwfZdXrLEL/61jhSWBLgsAAEsg3ARITCunXpt4mRKiQvTVD4W6Y9EWFbjKAl0WAAAtHuEmgC5oHarXJvZXm7Ag7f42T794jQdtAgBwrgg3AdY5NkKL7+ivsGC7PjlwVFPf5EGbAACcC8JNM9ArsbVeLn/Q5vuf5uhX7/KgTQAA6otw00xUftDmG1sO6dkPedAmAAD1QbhpRoZfnKCnfup90Ob8tV9pwXoetAkAQF0RbpqZW/snafpw74M2n1rxhf6+jQdtAgBQF4SbZijt6hTdU/GgzQwetAkAQF0QbpohwzA0Y0Q33ZzawfugzTd2atNXRwNdFgAALQLhppkyDENP/fQSDe0ep5Iyj+5+dZv2fpsX6LIAAGj2CDfNmMNu09zb+ujylGgVuMo0ftEWHrQJAMBZEG6auZAgu14el6qLL4jUscISjVu4Rdl5PGgTAIAzIdy0AN4HbfZXSttwHT5+UmMXbtGPPGgTAIBqEW5aiLatnHrtLu+DNg/kFmjC4q08aBMAgGoQbloQvwdtHjqutNe286BNAABOQ7hpYTrHRmhR+YM2Pz5wRA8u5UGbAABURrhpgXonttaCsd4Hba7Ym6NfvfspD9oEAKAc4aaFurJLW/3h1ooHbWZpxrK92n7wmErKPIEuDQCAgDLM8+y//Pn5+YqKilJeXp4iIyMDXc45qwg2FUKD7OrbsY0uT4nWZSkx6tkhSk6HPYAVAgBw7ury/e1ooprQSG7rn6Q2YcH6x67D+nfmMR0rLNHHB47o4wNHJEkhQTZdmtRGl6fE6LLkaPVOak3YAQBYGkduLMTjMXXghwJt/vqo/v31MW3++qiOnnY/HKfDG3YuS4nW5Skx6p3YWiFBhB0AQPNWl+9vwk1Dyt4ttesmOZwNu916Mk1TB3ILtDnzWHngOaojBf5hJ9hhU5/E1rosJUaXp0Tr0qQ2hB0AQLNDuKlBo4WbkkLp6Y6SzS4lXS4lXyUlXyMl9JLszePsn2ma+uqHQm/QKQ88P5xw+fUJttvUO7G1b8zOpUltFBpM2AEABBbhpgaNFm6+/0x67adSwff+7c5IqeMV3rCTcrXU7iLJ1jwuUjNNU5lHCrW5/BTW5q+PKve0sBNkN9SrQ2vvmJ2UaPXt2EZhwc0jrAEAzh+Emxo06mkp05R+2C9lrpcy10nfbJCK8/z7hLWVkgeVH9m5WopOkQyjYeuoJ9M09c3RIt8prM1fH1NOfrFfH4fNUK/E1rosuXzMTlJrRYYEBahiAMD5gnBTgyYdUOxxSzl7ysPOeungRqm0yL9PZIdTR3WSr5Ii2zduTXVgmqayjhWVH9XxHt3Jziuu0i86PFhJ0WHqGBOmpOiw8vlwdYwJU2yEU0YzCW8AgJaLcFODgF4tVVYiHd5+6sjOoS2Sp9S/T0znU0d1Og2SwmOatsYamKapQ8dOesNOpveKrMPHT9a4TkiQTYltKoJP+KkAFBOmDm1CuSwdAFArhJsaNKtLwUuKpEObpa/XeQNP9i7JPO0Ow3GXnDqqkzRACmlel6+fKC5V1rEiZR0t0sFjRTp4tEiHjhXp4LFCHf7xpGp67JVhSO2jQn1He5JivCGoY3S4kqLDFBXG6S4AgFeLCTfr16/X7373O23fvl3Z2dl65513NGrUqBrXWbdundLT0/XZZ5+pffv2evjhh5WWllbrz2xW4eZ0J497T11lloed3M/9lxt26YJLvUd1kq+SEvtLQaEBKbU2St0eHf7xpLKOeYNP1tFCHTxa5A1Dx4pUVFLzE82jQoN8R3pOnfLyHv2JjwyRzcbpLgA4X7SYOxQXFhaqV69euuOOOzR69Oiz9s/MzNTIkSN19913a8mSJfrkk0903333qV27drVav9kLbS11G+mdJKkg99R4ncz10o+Z0rdbvdOG30t2pzfgpFwtxffyrh/S+tSrIzhguyJJQXabOrUNV6e24VWWmaapIwUlyjpW6A0/R08d/ck6VqQfTriUd7JUe77N055v86qs77AZio1wKjYyRPGRIYqL9M7Hlc/HR4YoNjJEkSEOxvwAwHmm2ZyWMgzjrEdupk+fruXLl2vfvn2+trS0NO3evVubNm2q1ec06yM3Z3M861TQ+XqdVJBTc39H6GmBJ8o//IREVQ1EFW1BYQG9iquopMwXeg6Vv1Yc/fn2x5Mqq+l8VyUhQTZv4IkIUVxUiOIinIqLDFFseQCKK5+4lw8ANG8t5shNXW3atElDhw71axs2bJgWLlyo0tJSBQVZfIxG6ySpz+3eyTSlI1+eOoX14zdS8XHpZJ7kKj/SUXZSOnFSOpFd98+yBdUuEDkjvEHIEeJ9DQr1nxyh3js21zEohQU71C0+Ut3iq/4Auz2mck8U6/t8l77PL1ZufrFy8iu/d+n7E8U6XlSq4lKPNxgdLarmU06JCHH4wk5spLM8EDkVHxXiOyLUrpVTwY7mcY8iAMCZtahwk5OTo7i4OL+2uLg4lZWV6ciRI0pISKiyjsvlkst16sZ0+fn5jV5nkzAMqd2F3qn/3f7LPG7Jle8dw1N8vPw1r9J8+fszLTfd3qu4Cn/wTude7JmDT1Bo+bKQU/N+QaliWfl7R4jsQWFKcDiVEOyU4oKlBIdkj5DsMZI9SLIHS/ZgFbul3BMl+v5Esb7PL1ZOXrFyT3gD0PflYSgnr1gnS906UVymE8UF+jK3oMY9iQoNUiunQ2HBdoU5HQoPtiss2KFwZ/lrsP3My3x9HApz2hUe7FBIkI3TZgDQwFpUuJFU5Yug4qzamb4gZs+erccff7zR62pWbHYptI13qivTlEoKag4/lcOR64T3CFHp6VORNyR5NyqVFnqnJhQiQ0n2YCXZg/1Cj2++VZDMqGC5DYdK5FCxx65ij11FbruKygwVlNl0otRQfqlNeSWSy7TLVRKsgpJQnVCoTphhKlCojpihylSoCsxQFShUhQqRVLvAYhhSWNDZwpBd4U6HWocFqXVosPc1rPw1NEhRYUFcUg8AlbSocBMfH6+cHP9xJrm5uXI4HIqJqf5+MDNmzFB6errvfX5+vhITExu1zhbNMLynmpwRUlSHc9uWu9QbckqLy19PnjkIlVXqU+Oy4lNt7lLJXXLq9fR7BsmU3C7vdKbdlfcfgUNSWE37Uofs4JEhly1cJ21hKjLCVKhQnTBDlW+GKs8TqjyPUz+WhahA3pBUUBaqE2VhKijwvj9WKSSZqvk0mE0eBatUrYM9ahsiRYcainFK0SGmWgebauM0FRlkKirYVESQWxF2j1o53Ap3eBRqdyvIU+r98ykrKf+zrJh3lf+5lnp/Jgybd5Lh/77KssrtNS2rbnuV+9i8VwfayiffvKN83lZp3lHex1ZpvqLddlqfStv0rV8xb/POV9Tke1U1bdUs4wictZmm91Ydpsd7dLxi3m+q6FPNcp9KP0NSDT9rtemr6pdXnrfZvcMMbPbz6me0RYWbAQMG6P/+7//82lauXKnU1NQzjrdxOp1yOpvHU7rPO/YgyR7lHafTFEyzUuApOcP82ZafZb60yHtky3Wi0pQvFed75023bDIV6ilQqKdA0dXVaUiq5fCwEnu4XPZwlRjBMtylsntKZDdL5TBL5FCZHKr0S7OkfEKA1fSFdKZlNu/YtPJTr75TsY6QSm3VvNa0zK9PqHebjlDvv8um+pIzTe8k89SXf8W8u0Qqc3kn33zxqfka28r/0+JrKz4VyqttKzn1H6KK4FEloFQKLzWGkxbMFuQN8vbTXn3zQd4HPdsc5fOnL6+8TkXfM2wzJEq6/N6A7WpAw01BQYEOHDjge5+Zmaldu3YpOjpaSUlJmjFjhg4fPqxXX31VkvfKqHnz5ik9PV133323Nm3apIULF+qNN94I1C6gOTEM7+XvgboE3jS9R5cqhx6/19Pmi/OrhqSKV0+ZJCnYXahgd+1P53lswXLbglVmC1KZglQqh0oUJJccKvZ4p5Plp95c5ctKTIdK5ZBLQSqp1FaiIJXJJu/Xrylb+WTII0Peo0Y2mbIZ3i8s7/KKZd5+3lez0qv/uqq0XZvhnewyFWSYCrabCrZJwTZTTptHwTYpyOZRkOGRw/D2cRhuOQxTdnlkl1t2mbIbHtnMMhmmR0bFl5inrHwsWfnkmy+rdPq0wX4Qyr/ET72tlZITDVzHGRg2/7AjqdrwUe18DWGluvnzlnHqaGLlI5SV/1wq/5l6G84831A8pd6prOY7yzeIVvHnb7jZtm2brr32Wt/7itNH48eP1+LFi5Wdna2srCzf8uTkZK1YsUIPPvig/vSnP6l9+/aaO3euNe5xg5bPMKTgMO8UEXf2/mdimt7/ZVYOPaXF3tBmd3rHC1XMO5x+Y4lshiGbzn5gyOMxdcJVpryiUh0/WaLjRaX6sahEeSdL5SoqVWFRqY4XlSi/uEwe01SZx5Tb41GZ25TbY8ptel8r3pd5PKfa3RX9T726Paf61fIq/gbRyulQK6dDESEOtQqpNO90qJUzSBEh5e+DbYpwSpHBdkWE2BUV4vC9BvluFmlW/UI64xdUDa9V1jNP/Z2XFvufui0rrt1rbfr4PtsTkDFwZ1T559j3GlL1Z9yvrdIyh/O0tvK+1bYF+4cOo/J8NadGbacvP73PmZY3wpGx03/mzjR/+s+i6fYesfKUnXr1zVe0l3nnq13uPjXvLn/vmy8tX7es0nz5NoJbNfyfQR00m/vcNJUWfZ8bwAI8lcJR5QDkC0geUyVlHhW63DrhKtWJ4jIVFJepwOWdvFe2lXrfF5fpRHlbgavU+764rNb3QaqNsGC7okKDFBkS5H0NDVJkqENRoUF+7VHlg7srv282V8OZpvdUTXWBSDr1pewbG1XTfF37Vx6TVKm9IpQ3hz8ftAiWvc8NgJbPZjNkk6GgRrrAyzRNuco85YGnIgDVLiTln/T2yz9ZqhMu76nBohK3ikrcys4rrnMtwXabIkMdiqwmCFUOSBEhQQoNsivYYZPTYSt/Pf29ty3IbtQ9MBnGqaMcwHmAcAPAUgzDUEiQXSFBdrWLqP+XeZnbG5DyTpYqv7hUeSe9U/7JslPzxRVt3ulUe5n3CJTboyMFJTpS0LAjvSsHIGc1Aej0UHR6WKqY9012Q8EOm4Ls3inYbqv03lCQ3btekN2mIIe3zWm3K8jhXeaw1SNwAY2IcAMA1XDYbWoTHqw24XUfoG6apgpL3N6wU3R6ODoVhPIrwtPJUrnKPCop88hV5i5/9Zx6dftfreMqbz+hsoba3XNiGPKFooow5A1NFYHI8M1XtIcGe29kGe50qJWz/F5PzlP3dQovv9dTK6dDYU6HWpXf/DLIzl3CcXaEGwBoYIZh+AYzX9A69OwrnIWn/ChQjQGovL26Pq6y6td1lXlUWuZRqdsboErd3j6lbtM3X9Fe6jZ970vK/MOWacq7rKzxL5kOdth8AajibuGnwpA3EFW/zDsfGlR+F/Fg73xocD1P9aFZI9wAQDNnsxkKsXlPtTUHZvmA8BK3R6Vl5mnBqCIQmf7vy/wDVFGJW4WuMhW43Coq8Y6DKnK5VVhSpkJXmQpdbm9biXe+4uhVRYj6sej0m3bWn91mKKw86IQG232hJ8w376iyPOy0+ZAgb2A6NX9qO05HMxlYfh4h3AAA6sQwDDnshhx2m9REt5XyBqIyFfpCkTcMnQpAVYNSQUmZiioFpcKSMp0scetkiVtFpW65y6+qc5ffGqFiEHlDsxmSw2aTzSbZDUM2myG7zTg1b3jf+y2vaPMtM2Q35Nfmt9wwZLedttwwfKcFT502PHV6MOj0sVbl46+C/PoZCq40vurUNk6tG2y3yWZrXuGNcAMAaPa8g5+D1brG56TUTUmZRydLvWHnZKk3GJ2ad6u4/NV/uUcnS739isrbK+Z9/cvbKo42eUx55xv6fpHNiN1m+A0+bxcRovcfGBSwegg3AIDzUsXVYlGhtXweSh2VuU+Fp4r7OXnMyq/ya3Obpvc+UL55VdN26j5RFduo2nZqvmL8VOXxVBWnCSvGUrl88xWnEU2/04jeedPv1OPp95Kq+MziUo9OyHukKpAINwAANAKH3aYIu00RIY0TngLJ4zFV6ikffF5poHlFYAo0wg0AAKgTm82Q02aX0yGpGd4bkhsGAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASyHcAAAASwl4uHnxxReVnJyskJAQ9e3bVxs2bDhj37Vr18owjCrTF1980YQVAwCA5iyg4Wbp0qWaOnWqZs2apZ07d2rQoEEaMWKEsrKyalxv//79ys7O9k1dunRpoooBAEBzF9Bw8/zzz2vixIm66667dNFFF2nOnDlKTEzU/Pnza1wvNjZW8fHxvslutzdRxQAAoLkLWLgpKSnR9u3bNXToUL/2oUOHauPGjTWu26dPHyUkJGjw4MFas2ZNjX1dLpfy8/P9JgAAYF0BCzdHjhyR2+1WXFycX3tcXJxycnKqXSchIUELFixQRkaGli1bpq5du2rw4MFav379GT9n9uzZioqK8k2JiYkNuh8AAKB5cQS6AMMw/N6bplmlrULXrl3VtWtX3/sBAwbo0KFD+v3vf6+rrrqq2nVmzJih9PR03/v8/HwCDgAAFhawIzdt27aV3W6vcpQmNze3ytGcmlx++eX68ssvz7jc6XQqMjLSbwIAANYVsHATHBysvn37atWqVX7tq1at0sCBA2u9nZ07dyohIaGhywMAAC1UQE9Lpaena+zYsUpNTdWAAQO0YMECZWVlKS0tTZL3lNLhw4f16quvSpLmzJmjTp06qUePHiopKdGSJUuUkZGhjIyMQO4GAABoRgIabm655RYdPXpUTzzxhLKzs3XxxRdrxYoV6tixoyQpOzvb7543JSUlmjZtmg4fPqzQ0FD16NFD7733nkaOHBmoXQAAAM2MYZqmGegimlJ+fr6ioqKUl5fH+BsAAFqIunx/B/zxCwAAAA0p4JeCAwBaHrfbrdLS0kCXAYsJDg6WzXbux10INwCAWjNNUzk5OTp+/HigS4EF2Ww2JScnKzg4+Jy2Q7gBANRaRbCJjY1VWFjYGW+6CtSVx+PRd999p+zsbCUlJZ3TzxbhBgBQK2632xdsYmJiAl0OLKhdu3b67rvvVFZWpqCgoHpvhwHFAIBaqRhjExYWFuBKYFUVp6Pcbvc5bYdwAwCoE05FobE01M8W4QYAAFgK4QYAgDq65pprNHXq1Fr3/+abb2QYhnbt2tVoNeEUwg0AwLIMw6hxmjBhQr22u2zZMv32t7+tdf/ExETfY4YaEyHKi6ulAACWlZ2d7ZtfunSpfv3rX2v//v2+ttDQUL/+paWltbpKJzo6uk512O12xcfH12kd1B9HbgAA9WaapopKypp8qu1jEePj431TVFSUDMPwvS8uLlbr1q3197//Xddcc41CQkK0ZMkSHT16VLfddps6dOigsLAwXXLJJXrjjTf8tnv6aalOnTrpqaee0p133qmIiAglJSVpwYIFvuWnH1FZu3atDMPQv/71L6WmpiosLEwDBw70C16S9L//+7+KjY1VRESE7rrrLj3yyCPq3bt3vf6uJMnlcun+++9XbGysQkJCdOWVV2rr1q2+5T/++KPGjBmjdu3aKTQ0VF26dNGiRYskeR9ePXnyZCUkJCgkJESdOnXS7Nmz611LY+LIDQCg3k6WutX91x82+ed+/sQwhQU3zFfY9OnT9dxzz2nRokVyOp0qLi5W3759NX36dEVGRuq9997T2LFjlZKSossuu+yM23nuuef029/+VjNnztTbb7+te++9V1dddZW6det2xnVmzZql5557Tu3atVNaWpruvPNOffLJJ5Kkv/3tb3ryySf14osv6oorrtCbb76p5557TsnJyfXe14cfflgZGRn661//qo4dO+rZZ5/VsGHDdODAAUVHR+vRRx/V559/rvfff19t27bVgQMHdPLkSUnS3LlztXz5cv39739XUlKSDh06pEOHDtW7lsZEuAEAnNemTp2qm266ya9t2rRpvvkpU6bogw8+0FtvvVVjuBk5cqTuu+8+Sd7A9MILL2jt2rU1hpsnn3xSV199tSTpkUce0fXXX6/i4mKFhIToj3/8oyZOnKg77rhDkvTrX/9aK1euVEFBQb32s7CwUPPnz9fixYs1YsQISdLLL7+sVatWaeHChXrooYeUlZWlPn36KDU1VZL3iFSFrKwsdenSRVdeeaUMw1DHjh3rVUdTINwAAOotNMiuz58YFpDPbSgVX+QV3G63nn76aS1dulSHDx+Wy+WSy+VSeHh4jdvp2bOnb77i9Fdubm6t10lISJAk5ebmKikpSfv37/eFpQr9+/fXRx99VKv9Ot1XX32l0tJSXXHFFb62oKAg9e/fX/v27ZMk3XvvvRo9erR27NihoUOHatSoURo4cKAkacKECbruuuvUtWtXDR8+XP/1X/+loUOH1quWxka4AQDUm2EYDXZ6KFBODy3PPfecXnjhBc2ZM0eXXHKJwsPDNXXqVJWUlNS4ndMHIhuGIY/HU+t1Km5gV3md029qV9uxRtWpWLe6bVa0jRgxQgcPHtR7772n1atXa/DgwZo0aZJ+//vf69JLL1VmZqbef/99rV69WjfffLOGDBmit99+u941NRYGFAMAUMmGDRt044036vbbb1evXr2UkpKiL7/8ssnr6Nq1q7Zs2eLXtm3btnpvr3PnzgoODtbHH3/saystLdW2bdt00UUX+dratWunCRMmaMmSJZozZ47fwOjIyEjdcsstevnll7V06VJlZGTo2LFj9a6psbTsuA0AQAPr3LmzMjIytHHjRrVp00bPP/+8cnJy/AJAU5gyZYruvvtupaamauDAgVq6dKn27NmjlJSUs657+lVXktS9e3fde++9euihhxQdHa2kpCQ9++yzKioq0sSJEyV5x/X07dtXPXr0kMvl0j//+U/ffr/wwgtKSEhQ7969ZbPZ9NZbbyk+Pl6tW7du0P1uCIQbAAAqefTRR5WZmalhw4YpLCxM99xzj0aNGqW8vLwmrWPMmDH6+uuvNW3aNBUXF+vmm2/WhAkTqhzNqc6tt95apS0zM1NPP/20PB6Pxo4dqxMnTig1NVUffvih2rRpI8n74MoZM2bom2++UWhoqAYNGqQ333xTktSqVSs988wz+vLLL2W329WvXz+tWLFCNlvzOwlkmOdyAq8Fys/PV1RUlPLy8hQZGRnocgCgxSguLlZmZqaSk5MVEhIS6HLOS9ddd53i4+P12muvBbqURlHTz1hdvr85cgMAQDNUVFSkl156ScOGDZPdbtcbb7yh1atXa9WqVYEurdkj3AAA0AwZhqEVK1bof//3f+VyudS1a1dlZGRoyJAhgS6t2SPcAADQDIWGhmr16tWBLqNFan6jgAAAAM4B4QYAAFgK4QYAAFgK4QYAAFgK4QYAAFgK4QYAAFgK4QYAgLO45pprNHXqVN/7Tp06ac6cOTWuYxiG3n333XP+7IbazvmEcAMAsKwbbrjhjDe927RpkwzD0I4dO+q83a1bt+qee+451/L8PPbYY+rdu3eV9uzsbI0YMaJBP+t0ixcvbpYPwKwvwg0AwLImTpyojz76SAcPHqyy7JVXXlHv3r116aWX1nm77dq1U1hYWEOUeFbx8fFyOp1N8llWQbgBANSfaUolhU0/1fKZz//1X/+l2NhYLV682K+9qKhIS5cu1cSJE3X06FHddttt6tChg8LCwnTJJZfojTfeqHG7p5+W+vLLL3XVVVcpJCRE3bt3r/b5T9OnT9eFF16osLAwpaSk6NFHH1Vpaakk75GTxx9/XLt375ZhGDIMw1fz6ael9u7dq5/85CcKDQ1VTEyM7rnnHhUUFPiWT5gwQaNGjdLvf/97JSQkKCYmRpMmTfJ9Vn1kZWXpxhtvVKtWrRQZGambb75Z33//vW/57t27de211yoiIkKRkZHq27evtm3bJkk6ePCgbrjhBrVp00bh4eHq0aOHVqxYUe9aaoPHLwAA6q+0SHqqfdN/7szvpODws3ZzOBwaN26cFi9erF//+tcyDEOS9NZbb6mkpERjxoxRUVGR+vbtq+nTpysyMlLvvfeexo4dq5SUFF122WVn/QyPx6ObbrpJbdu21ebNm5Wfn+83PqdCRESEFi9erPbt22vv3r26++67FRERoYcffli33HKLPv30U33wwQe+Ry5ERUVV2UZRUZGGDx+uyy+/XFu3blVubq7uuusuTZ482S/ArVmzRgkJCVqzZo0OHDigW265Rb1799bdd9991v05nWmaGjVqlMLDw7Vu3TqVlZXpvvvu0y233KK1a9dKksaMGaM+ffpo/vz5stvt2rVrl4KCgiRJkyZNUklJidavX6/w8HB9/vnnatWqVZ3rqAvCDQDA0u6880797ne/09q1a3XttddK8p6Suummm9SmTRu1adNG06ZN8/WfMmWKPvjgA7311lu1CjerV6/Wvn379M0336hDhw6SpKeeeqrKOJlf/epXvvlOnTrpl7/8pZYuXaqHH35YoaGhatWqlRwOh+Lj48/4WX/729908uRJvfrqqwoP94a7efPm6YYbbtAzzzyjuLg4SVKbNm00b9482e12devWTddff73+9a9/1SvcrF69Wnv27FFmZqYSExMlSa+99pp69OihrVu3ql+/fsrKytJDDz2kbt26SZK6dOniWz8rK0ujR4/WJZdcIklKSUmpcw11RbgBANRfUJj3KEogPreWunXrpoEDB+qVV17Rtddeq6+++kobNmzQypUrJUlut1tPP/20li5dqsOHD8vlcsnlcvnCw9ns27dPSUlJvmAjSQMGDKjS7+2339acOXN04MABFRQUqKysTJGRkbXej4rP6tWrl19tV1xxhTwej/bv3+8LNz169JDdbvf1SUhI0N69e+v0WZU/MzEx0RdsJKl79+5q3bq19u3bp379+ik9PV133XWXXnvtNQ0ZMkT/8z//o//3//6fJOn+++/Xvffeq5UrV2rIkCEaPXq0evbsWa9aaosxNwCA+jMM7+mhpp7KTy/V1sSJE5WRkaH8/HwtWrRIHTt21ODBgyVJzz33nF544QU9/PDD+uijj7Rr1y4NGzZMJSUltdq2Wc34H+O0+jZv3qxbb71VI0aM0D//+U/t3LlTs2bNqvVnVP6s07dd3WdWnBKqvMzj8dTps872mZXbH3vsMX322We6/vrr9dFHH6l79+565513JEl33XWXvv76a40dO1Z79+5Vamqq/vjHP9arltoi3AAALO/mm2+W3W7X66+/rr/+9a+64447fF/MGzZs0I033qjbb79dvXr1UkpKir788stab7t79+7KysrSd9+dOoK1adMmvz6ffPKJOnbsqFmzZik1NVVdunSpcgVXcHCw3G73WT9r165dKiws9Nu2zWbThRdeWOua66Ji/w4dOuRr+/zzz5WXl6eLLrrI13bhhRfqwQcf1MqVK3XTTTdp0aJFvmWJ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",
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
    "#plt.xticks(np.arange(0, +1, 5.0))\n",
    "plt.legend(loc='lower right')\n",
    "\n",
    "validation_loss, validation_acc = model.evaluate(X_val_scaled,  y_val_enc, verbose=2)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "cf3c14db-15f8-4269-bbeb-9582e2771789",
   "metadata": {},
   "source": [
    "### Predict and evaluate performance on test set (out-of-sample)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "20aff460-7cc0-4ed2-9d70-d77cc6d0e77b",
   "metadata": {},
   "source": [
    "**Predict on the test set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6c59ee48-1608-418f-b5ac-4098b16b21c7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:46:39.403352Z",
     "iopub.status.busy": "2023-05-16T19:46:39.402892Z",
     "iopub.status.idle": "2023-05-16T19:46:45.122359Z",
     "shell.execute_reply": "2023-05-16T19:46:45.120952Z",
     "shell.execute_reply.started": "2023-05-16T19:46:39.403319Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3571/3571 [==============================] - 4s 1ms/step\n"
     ]
    }
   ],
   "source": [
    "# Predict the labels for the test set\n",
    "y_pred_prob = model.predict(X_test_scaled)\n",
    "\n",
    "# Convert probabilities to class labels\n",
    "y_pred = np.argmax(y_pred_prob, axis=1)\n",
    "y_test_labels = np.argmax(y_test_enc, axis=1)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "282aec36-6a3a-410f-9714-d44db105cce2",
   "metadata": {},
   "source": [
    "**Confusion matrix**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "fd030671-d114-4cb1-94c3-e84ebbfc1f4b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:46:47.501350Z",
     "iopub.status.busy": "2023-05-16T19:46:47.500677Z",
     "iopub.status.idle": "2023-05-16T19:46:47.530742Z",
     "shell.execute_reply": "2023-05-16T19:46:47.528599Z",
     "shell.execute_reply.started": "2023-05-16T19:46:47.501303Z"
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
      "| Class 0 |  31076  |   1738  |   2053  |   3848  |   1535  |   775   |\n",
      "| Class 1 |   2689  |  23581  |   1239  |   2139  |   1646  |   819   |\n",
      "| Class 2 |   716   |   438   |   4633  |   581   |   252   |   498   |\n",
      "| Class 3 |   3901  |   1521  |   1373  |  15668  |   1279  |   452   |\n",
      "| Class 4 |   1459  |   793   |   835   |   1502  |   2182  |   712   |\n",
      "| Class 5 |   168   |   131   |   375   |   162   |   178   |   1309  |\n",
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
   "id": "eb2d4713-3c27-4c2b-ab86-a21fd15cb270",
   "metadata": {},
   "source": [
    "**Classification report**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "6e1ad0b2-c644-443d-96be-89c2eedf3742",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-16T19:46:52.418908Z",
     "iopub.status.busy": "2023-05-16T19:46:52.418272Z",
     "iopub.status.idle": "2023-05-16T19:46:52.623211Z",
     "shell.execute_reply": "2023-05-16T19:46:52.622403Z",
     "shell.execute_reply.started": "2023-05-16T19:46:52.418861Z"
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
      "|      0       |    0.78   |  0.76  |   0.77   |  41025  |\n",
      "|      1       |    0.84   |  0.73  |   0.78   |  32113  |\n",
      "|      2       |    0.44   |  0.65  |   0.53   |   7118  |\n",
      "|      3       |    0.66   |  0.65  |   0.65   |  24194  |\n",
      "|      4       |    0.31   |  0.29  |   0.3    |   7483  |\n",
      "|      5       |    0.29   |  0.56  |   0.38   |   2323  |\n",
      "|  macro avg   |    0.55   |  0.61  |   0.57   |         |\n",
      "| weighted avg |    0.71   |  0.69  |   0.69   |         |\n",
      "+--------------+-----------+--------+----------+---------+\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "from prettytable import PrettyTable\n",
    "\n",
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
    "print(table)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1fcc160-404a-4ff8-a326-20c0259a1884",
   "metadata": {},
   "outputs": [],
   "source": []
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
