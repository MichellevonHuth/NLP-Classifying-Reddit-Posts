{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fd37f26f-1996-4b04-b001-a511e875bbbd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:21:48.703656Z",
     "iopub.status.busy": "2023-05-21T09:21:48.702860Z",
     "iopub.status.idle": "2023-05-21T09:22:06.681061Z",
     "shell.execute_reply": "2023-05-21T09:22:06.675186Z",
     "shell.execute_reply.started": "2023-05-21T09:21:48.703601Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "!pip install scikit-learn prettytable seaborn --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b0855c06",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:06.699397Z",
     "iopub.status.busy": "2023-05-21T09:22:06.699191Z",
     "iopub.status.idle": "2023-05-21T09:22:09.240121Z",
     "shell.execute_reply": "2023-05-21T09:22:09.239325Z",
     "shell.execute_reply.started": "2023-05-21T09:22:06.699380Z"
    },
    "id": "b0855c06",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# imports \n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import pickle\n",
    "from prettytable import PrettyTable\n",
    "\n",
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix, classification_report"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e48Gr5wuKIox",
   "metadata": {
    "id": "e48Gr5wuKIox"
   },
   "source": [
    "# Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bd6f8bce",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:09.242308Z",
     "iopub.status.busy": "2023-05-21T09:22:09.242015Z",
     "iopub.status.idle": "2023-05-21T09:22:59.373047Z",
     "shell.execute_reply": "2023-05-21T09:22:59.372178Z",
     "shell.execute_reply.started": "2023-05-21T09:22:09.242288Z"
    },
    "id": "bd6f8bce",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load pickled pre-processed data\n",
    "# Load sparse\n",
    "X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse, org_train, org_test = pd.read_pickle(\"pickles/sparse.pkl\")\n",
    "\n",
    "# Load dense\n",
    "X_train_dense, X_test_dense, y_train_dense, y_test_dense, w2v_train, w2v_test = pd.read_pickle(\"pickles/dense.pkl\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c044170a-3b4c-4dc1-8c51-950d593a3b1b",
   "metadata": {},
   "source": [
    "# Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0bc15ac1-04f4-4a35-bb8d-c9c595768f46",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:59.374113Z",
     "iopub.status.busy": "2023-05-21T09:22:59.373913Z",
     "iopub.status.idle": "2023-05-21T09:22:59.379727Z",
     "shell.execute_reply": "2023-05-21T09:22:59.379112Z",
     "shell.execute_reply.started": "2023-05-21T09:22:59.374096Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Function to print out formatted confusion matrix\n",
    "def pretty_confusion_matrix(y_test, y_pred, header_to_print):\n",
    "    # Generate confusion matrix\n",
    "    confmat = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "    # Instatiate pretty table\n",
    "    tab = PrettyTable()\n",
    "    tab.field_names = [\"\", \"Predicted 0\", \"Predicted 1\", \"Predicted 2\", \"Predicted 3\", \"Predicted 4\", \"Predicted 5\"]\n",
    "    \n",
    "    # Fill in rows and print table\n",
    "    for i in range(6):\n",
    "        tab.add_row([\"Actual \" + str(i)] + [confmat[i][j] for j in range(6)])\n",
    "    print(f\"Confusion Matrix ({header_to_print}): \\n{tab}\")\n",
    "    \n",
    "    # Print a legend of the classes\n",
    "    classes = y_test.unique()\n",
    "    print(\"0: BPD, 1: Anxiety, 2: Bipolar, 3: Depression, 4: Mental Illness, 5: Schizophrenia\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "47b92541-75b4-442f-868d-b43ffb033b93",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:59.380671Z",
     "iopub.status.busy": "2023-05-21T09:22:59.380486Z",
     "iopub.status.idle": "2023-05-21T09:22:59.562182Z",
     "shell.execute_reply": "2023-05-21T09:22:59.560816Z",
     "shell.execute_reply.started": "2023-05-21T09:22:59.380654Z"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Function to print out formatted classification report\n",
    "def pretty_classification_report(y_test, y_pred, header_to_print):\n",
    "    # Generate classification report\n",
    "    class_report = classification_report(y_test, y_pred, output_dict=True)\n",
    "\n",
    "    # Instatiate the PrettyTable\n",
    "    tab = PrettyTable()\n",
    "    tab.field_names = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']\n",
    "\n",
    "    # Get a list of classes\n",
    "    classes = y_test_sparse.unique()\n",
    "\n",
    "    # Loop through each class in the classification report and add its metrics to the table\n",
    "    for class_name, metrics in class_report.items():\n",
    "        if class_name in classes:\n",
    "            class_id = class_name\n",
    "            precision = round(metrics['precision'], 2)\n",
    "            recall = round(metrics['recall'], 2)\n",
    "            f1_score = round(metrics['f1-score'], 2)\n",
    "            support = metrics['support']\n",
    "            tab.add_row([class_id, precision, recall, f1_score, support])\n",
    "\n",
    "    # Add the macro and weighted averages to the table\n",
    "    macro_precision = round(class_report['macro avg']['precision'], 2)\n",
    "    macro_recall = round(class_report['macro avg']['recall'], 2)\n",
    "    macro_f1_score = round(class_report['macro avg']['f1-score'], 2)\n",
    "    tab.add_row(['macro avg', macro_precision, macro_recall, macro_f1_score, ''])\n",
    "\n",
    "    weighted_precision = round(class_report['weighted avg']['precision'], 2)\n",
    "    weighted_recall = round(class_report['weighted avg']['recall'], 2)\n",
    "    weighted_f1_score = round(class_report['weighted avg']['f1-score'], 2)\n",
    "    tab.add_row(['weighted avg', weighted_precision, weighted_recall, weighted_f1_score, ''])\n",
    "\n",
    "    # Print the table\n",
    "    print(f\"Classification Report ({header_to_print}): \\n{tab}\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "aca97ec9-f4e9-4fe8-bdac-6be3495d00d7",
   "metadata": {
    "id": "P70gR7g9KDge",
    "tags": []
   },
   "source": [
    "# Word2Vec"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "674b60b9-6bd1-4bef-820a-5db725708f82",
   "metadata": {
    "id": "_MwskHiuKq_X"
   },
   "source": [
    "**Using grid search for hyperparameter tuning**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fff86d9b-05f5-45ee-8082-83cd39eafddb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:59.564831Z",
     "iopub.status.busy": "2023-05-21T09:22:59.564266Z",
     "iopub.status.idle": "2023-05-21T09:22:59.939103Z",
     "shell.execute_reply": "2023-05-21T09:22:59.937730Z",
     "shell.execute_reply.started": "2023-05-21T09:22:59.564780Z"
    },
    "id": "nCJxWESIKM-t",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# set max_iter = 1000 as below the max_iter was reached which means the coef_ did not converge\n",
    "# solver='saga' as it was the best solver in previous grid searchs, saga is well suited for large datasets\n",
    "lr_w2v = LogisticRegression(multi_class = 'multinomial', max_iter=1000, solver='saga')\n",
    "\n",
    "# Define the hyperparameter grid to search\n",
    "parameter_space = {\n",
    "    'penalty': ['l1', 'l2'], # Regularization penalty (L1 or L2)\n",
    "    'C': [1, 10, 100], # Regularization strength\n",
    "    'tol': [1e-5, 1e-4, 1e-3], # Tolerance for stopping criteria\n",
    "}\n",
    "\n",
    "# Perform grid search with cross-validation\n",
    "lr_clf_w2v = GridSearchCV(lr_w2v, parameter_space, n_jobs=-1, cv=3, verbose=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "03d22dc0-f7cd-4d9f-9d40-d93579844bce",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 227
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T09:22:59.941937Z",
     "iopub.status.busy": "2023-05-21T09:22:59.941234Z",
     "iopub.status.idle": "2023-05-21T09:51:20.495169Z",
     "shell.execute_reply": "2023-05-21T09:51:20.494246Z",
     "shell.execute_reply.started": "2023-05-21T09:22:59.941885Z"
    },
    "id": "l6lnwdiiK9cq",
    "outputId": "3ba05e72-050a-46e0-d0f8-f0ed3d7bb2a9",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 18 candidates, totalling 54 fits\n",
      "CPU times: user 5min 22s, sys: 3.16 s, total: 5min 25s\n",
      "Wall time: 28min 20s\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(max_iter=1000,\n",
       "                                          multi_class=&#x27;multinomial&#x27;,\n",
       "                                          solver=&#x27;saga&#x27;),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;C&#x27;: [1, 10, 100], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],\n",
       "                         &#x27;tol&#x27;: [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(max_iter=1000,\n",
       "                                          multi_class=&#x27;multinomial&#x27;,\n",
       "                                          solver=&#x27;saga&#x27;),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;C&#x27;: [1, 10, 100], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],\n",
       "                         &#x27;tol&#x27;: [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(max_iter=1000, multi_class=&#x27;multinomial&#x27;, solver=&#x27;saga&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(max_iter=1000, multi_class=&#x27;multinomial&#x27;, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(max_iter=1000,\n",
       "                                          multi_class='multinomial',\n",
       "                                          solver='saga'),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'C': [1, 10, 100], 'penalty': ['l1', 'l2'],\n",
       "                         'tol': [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "lr_clf_w2v.fit(X_train_dense, y_train_dense)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4377aed6-bbbc-4910-ab9e-3107b4f4c2ed",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T09:51:20.496968Z",
     "iopub.status.busy": "2023-05-21T09:51:20.496750Z",
     "iopub.status.idle": "2023-05-21T09:51:20.503151Z",
     "shell.execute_reply": "2023-05-21T09:51:20.502425Z",
     "shell.execute_reply.started": "2023-05-21T09:51:20.496949Z"
    },
    "id": "t8zVMpxDLnim",
    "outputId": "fdf24f49-cd1b-4eec-bafa-5041bbc36ba5",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best hyperparameters: \n",
      " {'C': 10, 'penalty': 'l1', 'tol': 0.0001}\n",
      "Best score: \n",
      " 0.651770015592991\n"
     ]
    }
   ],
   "source": [
    "# Print the best hyperparameters and the corresponding score\n",
    "print(\"Best hyperparameters: \\n\", lr_clf_w2v.best_params_)\n",
    "print(\"Best score: \\n\", lr_clf_w2v.best_score_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d9b8d23d-9bbc-481e-9416-4ac976786cb5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T09:51:20.507925Z",
     "iopub.status.busy": "2023-05-21T09:51:20.507739Z",
     "iopub.status.idle": "2023-05-21T10:22:44.642196Z",
     "shell.execute_reply": "2023-05-21T10:22:44.639804Z",
     "shell.execute_reply.started": "2023-05-21T09:51:20.507909Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validated scores: [0.65121657 0.65365949 0.65091205 0.65299674 0.65100977]\n"
     ]
    }
   ],
   "source": [
    "# Save optimal model\n",
    "best_model_lr_w2v = lr_clf_w2v.best_estimator_\n",
    "\n",
    "# Perform cross validation to check if model is overfitting \n",
    "y_scores = cross_val_score(best_model_lr_w2v, X_train_dense, y_train_dense, cv = 5)\n",
    "print(f'Cross-validated scores: {y_scores}')"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "709f32f9-c75a-417c-aae4-b1808310c417",
   "metadata": {
    "id": "Unv7qMxyMPEi"
   },
   "source": [
    "**Evaluate the performance on the test set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "534d96a0-7b75-4e56-81f3-d0a5bf8102a2",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T10:22:44.646296Z",
     "iopub.status.busy": "2023-05-21T10:22:44.645473Z",
     "iopub.status.idle": "2023-05-21T10:22:44.695002Z",
     "shell.execute_reply": "2023-05-21T10:22:44.693511Z",
     "shell.execute_reply.started": "2023-05-21T10:22:44.646237Z"
    },
    "id": "srNUarSTMR1O",
    "outputId": "6a0d81b6-d2fe-4de8-fd00-2eff93708576",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Predict on the test set \n",
    "y_pred_lr_w2v = best_model_lr_w2v.predict(X_test_dense)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "b22328b4-4f88-4eae-92d6-510eca1fa2ad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T10:22:44.698473Z",
     "iopub.status.busy": "2023-05-21T10:22:44.697023Z",
     "iopub.status.idle": "2023-05-21T10:22:45.201292Z",
     "shell.execute_reply": "2023-05-21T10:22:45.199839Z",
     "shell.execute_reply.started": "2023-05-21T10:22:44.698423Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix (Logistic Regression, Word2Vec): \n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "|          | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 | Predicted 5 |\n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "| Actual 0 |    32692    |     1955    |     1067    |     3947    |     1094    |     270     |\n",
      "| Actual 1 |     3381    |    24535    |     742     |     2173    |     1004    |     278     |\n",
      "| Actual 2 |     1154    |     666     |     4024    |     693     |     326     |     255     |\n",
      "| Actual 3 |     4604    |     1661    |     708     |    15986    |     1058    |     177     |\n",
      "| Actual 4 |     2054    |     1011    |     492     |     1591    |     1960    |     375     |\n",
      "| Actual 5 |     293     |     239     |     287     |     221     |     296     |     987     |\n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "0: BPD, 1: Anxiety, 2: Bipolar, 3: Depression, 4: Mental Illness, 5: Schizophrenia\n"
     ]
    }
   ],
   "source": [
    "# Print confusion matrix\n",
    "pretty_confusion_matrix(y_test_dense, y_pred_lr_w2v, \"Logistic Regression, Word2Vec\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "42362496-2a60-40da-86ec-389b992d663e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T10:22:45.204414Z",
     "iopub.status.busy": "2023-05-21T10:22:45.203629Z",
     "iopub.status.idle": "2023-05-21T10:22:48.369111Z",
     "shell.execute_reply": "2023-05-21T10:22:48.368052Z",
     "shell.execute_reply.started": "2023-05-21T10:22:45.204361Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report (Logistic Regression, Word2Vec): \n",
      "+---------------+-----------+--------+----------+---------+\n",
      "|     Class     | Precision | Recall | F1-score | Support |\n",
      "+---------------+-----------+--------+----------+---------+\n",
      "|      BPD      |    0.74   |  0.8   |   0.77   |  41025  |\n",
      "|    anxiety    |    0.82   |  0.76  |   0.79   |  32113  |\n",
      "|    bipolar    |    0.55   |  0.57  |   0.56   |   7118  |\n",
      "|   depression  |    0.65   |  0.66  |   0.66   |  24194  |\n",
      "| mentalillness |    0.34   |  0.26  |   0.3    |   7483  |\n",
      "| schizophrenia |    0.42   |  0.42  |   0.42   |   2323  |\n",
      "|   macro avg   |    0.59   |  0.58  |   0.58   |         |\n",
      "|  weighted avg |    0.7    |  0.7   |   0.7    |         |\n",
      "+---------------+-----------+--------+----------+---------+\n"
     ]
    }
   ],
   "source": [
    "# Print out classification report\n",
    "pretty_classification_report(y_test_dense, y_pred_lr_w2v, \"Logistic Regression, Word2Vec\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "eae33144-2b00-4e84-bb64-9db5f23f37ae",
   "metadata": {
    "id": "yTlmfhu3P5fN"
   },
   "source": [
    "**Save the model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "20d34908-f8e6-4c41-8b20-3665b68c4fd0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T21:23:06.650177Z",
     "iopub.status.busy": "2023-05-17T21:23:06.649413Z",
     "iopub.status.idle": "2023-05-17T21:23:06.661648Z",
     "shell.execute_reply": "2023-05-17T21:23:06.660347Z",
     "shell.execute_reply.started": "2023-05-17T21:23:06.650125Z"
    },
    "id": "gNk51XF2QBaz",
    "tags": []
   },
   "outputs": [],
   "source": [
    "pickle.dump(best_model_lr_w2v, open('logistic_regression_model_w2v.pkl', 'wb'))"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "NeE30R8sKQb9",
   "metadata": {
    "id": "NeE30R8sKQb9"
   },
   "source": [
    "# TF-IDF"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f5d8b3b9",
   "metadata": {
    "id": "f5d8b3b9"
   },
   "source": [
    "**Using grid search for hyperparameter tuning**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b26d09d6-25a0-4635-bb81-5dabbffe179c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T10:22:48.370092Z",
     "iopub.status.busy": "2023-05-21T10:22:48.369893Z",
     "iopub.status.idle": "2023-05-21T10:22:48.374934Z",
     "shell.execute_reply": "2023-05-21T10:22:48.374345Z",
     "shell.execute_reply.started": "2023-05-21T10:22:48.370074Z"
    },
    "id": "b26d09d6-25a0-4635-bb81-5dabbffe179c",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# set max_iter = 1000 as below the max_iter was reached which means the coef_ did not converge\n",
    "# solver='saga' as it was the best solver in previous grid searchs\n",
    "lr_tfidf = LogisticRegression(multi_class = 'multinomial', max_iter=100, solver='saga')\n",
    "\n",
    "# Define the hyperparameter grid to search\n",
    "parameter_space = {\n",
    "    'penalty': ['l1', 'l2'], # Regularization penalty (L1 or L2)\n",
    "    'C': [1, 10, 100], # Regularization strength\n",
    "    'tol': [1e-5, 1e-4, 1e-3], # Tolerance for stopping criteria\n",
    "}\n",
    "\n",
    "# Perform grid search with cross-validation\n",
    "lr_clf_tfidf = GridSearchCV(lr_tfidf, parameter_space, n_jobs=-1, cv=3, verbose=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "08ad198e",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 227
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T10:22:48.375839Z",
     "iopub.status.busy": "2023-05-21T10:22:48.375668Z",
     "iopub.status.idle": "2023-05-21T12:04:20.560847Z",
     "shell.execute_reply": "2023-05-21T12:04:20.559923Z",
     "shell.execute_reply.started": "2023-05-21T10:22:48.375823Z"
    },
    "id": "08ad198e",
    "outputId": "bec38f0f-a97e-4c87-8024-b3357043c581",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 18 candidates, totalling 54 fits\n",
      "CPU times: user 3min 14s, sys: 3.25 s, total: 3min 18s\n",
      "Wall time: 1h 41min 31s\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(multi_class=&#x27;multinomial&#x27;,\n",
       "                                          solver=&#x27;saga&#x27;),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;C&#x27;: [1, 10, 100], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],\n",
       "                         &#x27;tol&#x27;: [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(multi_class=&#x27;multinomial&#x27;,\n",
       "                                          solver=&#x27;saga&#x27;),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;C&#x27;: [1, 10, 100], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],\n",
       "                         &#x27;tol&#x27;: [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(multi_class=&#x27;multinomial&#x27;, solver=&#x27;saga&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(multi_class=&#x27;multinomial&#x27;, solver=&#x27;saga&#x27;)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=3,\n",
       "             estimator=LogisticRegression(multi_class='multinomial',\n",
       "                                          solver='saga'),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'C': [1, 10, 100], 'penalty': ['l1', 'l2'],\n",
       "                         'tol': [1e-05, 0.0001, 0.001]},\n",
       "             verbose=10)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "lr_clf_tfidf.fit(X_train_sparse, y_train_sparse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1f6353ca-53bf-4afc-89c1-f3121357c48e",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T12:04:20.562814Z",
     "iopub.status.busy": "2023-05-21T12:04:20.562602Z",
     "iopub.status.idle": "2023-05-21T12:04:20.567749Z",
     "shell.execute_reply": "2023-05-21T12:04:20.567222Z",
     "shell.execute_reply.started": "2023-05-21T12:04:20.562793Z"
    },
    "id": "f4729f8e-2ba5-4a6f-bba3-530348d40eb8",
    "outputId": "aebf087c-2a32-403f-93be-fefe4fbd7810",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best hyperparameters: \n",
      " {'C': 1, 'penalty': 'l1', 'tol': 0.001}\n",
      "Best score: \n",
      " 0.727718002339203\n"
     ]
    }
   ],
   "source": [
    "# Print the best hyperparameters and the corresponding score\n",
    "print(\"Best hyperparameters: \\n\", lr_clf_tfidf.best_params_)\n",
    "print(\"Best score: \\n\", lr_clf_tfidf.best_score_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a4638cdd-8c56-4420-9cd5-e8ffe96d0d13",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T12:04:20.568576Z",
     "iopub.status.busy": "2023-05-21T12:04:20.568403Z",
     "iopub.status.idle": "2023-05-21T12:18:11.745961Z",
     "shell.execute_reply": "2023-05-21T12:18:11.744932Z",
     "shell.execute_reply.started": "2023-05-21T12:04:20.568559Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validated scores: [0.72921484 0.73002898 0.72370221 0.73187651 0.73151827]\n"
     ]
    }
   ],
   "source": [
    "# Save optimal model\n",
    "best_model_lr_tfidf = lr_clf_tfidf.best_estimator_\n",
    "\n",
    "# Perform cross validation to check if model is overfitting \n",
    "y_scores = cross_val_score(best_model_lr_tfidf, X_train_sparse, y_train_sparse, cv = 5)\n",
    "print(f'Cross-validated scores: {y_scores}')"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "73929de3-69cd-41e5-8da5-b64768681de3",
   "metadata": {
    "id": "50dfc909"
   },
   "source": [
    "**Evaluate the performance on the test set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3a9e172a-f057-4de3-aa1c-b51349e5c787",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "execution": {
     "iopub.execute_input": "2023-05-21T12:18:12.174245Z",
     "iopub.status.busy": "2023-05-21T12:18:12.174034Z",
     "iopub.status.idle": "2023-05-21T12:18:12.240749Z",
     "shell.execute_reply": "2023-05-21T12:18:12.239997Z",
     "shell.execute_reply.started": "2023-05-21T12:18:12.174227Z"
    },
    "id": "l5_C8BJRPLL7",
    "outputId": "7a80dedc-f015-4631-db8a-dcc2a73879dd",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Predict on the test set \n",
    "y_pred_lr_tfidf = best_model_lr_tfidf.predict(X_test_sparse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3fb73c10-c403-4936-954b-2139c4e2c8cd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T12:18:12.241903Z",
     "iopub.status.busy": "2023-05-21T12:18:12.241689Z",
     "iopub.status.idle": "2023-05-21T12:18:12.665218Z",
     "shell.execute_reply": "2023-05-21T12:18:12.664478Z",
     "shell.execute_reply.started": "2023-05-21T12:18:12.241885Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix (Logistic Regression, TF-IDF): \n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "|          | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 | Predicted 5 |\n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "| Actual 0 |    33956    |     1447    |     822     |     3284    |     1319    |     200     |\n",
      "| Actual 1 |     1605    |    26846    |     272     |     2006    |     1211    |     174     |\n",
      "| Actual 2 |     782     |     349     |     4844    |     593     |     328     |     222     |\n",
      "| Actual 3 |     2565    |     1494    |     403     |    18264    |     1369    |     103     |\n",
      "| Actual 4 |     1294    |     873     |     395     |     1545    |     3017    |     363     |\n",
      "| Actual 5 |     149     |     152     |     191     |     188     |     254     |     1392    |\n",
      "+----------+-------------+-------------+-------------+-------------+-------------+-------------+\n",
      "0: BPD, 1: Anxiety, 2: Bipolar, 3: Depression, 4: Mental Illness, 5: Schizophrenia\n"
     ]
    }
   ],
   "source": [
    "# Print confusion matrix\n",
    "pretty_confusion_matrix(y_test_sparse, y_pred_lr_tfidf, \"Logistic Regression, TF-IDF\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b0c4a38e-f6e8-40e3-8798-d4c3d683986b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T12:18:12.666387Z",
     "iopub.status.busy": "2023-05-21T12:18:12.666166Z",
     "iopub.status.idle": "2023-05-21T12:18:16.270963Z",
     "shell.execute_reply": "2023-05-21T12:18:16.270208Z",
     "shell.execute_reply.started": "2023-05-21T12:18:12.666369Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report (Logistic Regression, TF-IDF): \n",
      "+---------------+-----------+--------+----------+---------+\n",
      "|     Class     | Precision | Recall | F1-score | Support |\n",
      "+---------------+-----------+--------+----------+---------+\n",
      "|      BPD      |    0.84   |  0.83  |   0.83   |  41028  |\n",
      "|    anxiety    |    0.86   |  0.84  |   0.85   |  32114  |\n",
      "|    bipolar    |    0.7    |  0.68  |   0.69   |   7118  |\n",
      "|   depression  |    0.71   |  0.75  |   0.73   |  24198  |\n",
      "| mentalillness |    0.4    |  0.4   |   0.4    |   7487  |\n",
      "| schizophrenia |    0.57   |  0.6   |   0.58   |   2326  |\n",
      "|   macro avg   |    0.68   |  0.68  |   0.68   |         |\n",
      "|  weighted avg |    0.78   |  0.77  |   0.77   |         |\n",
      "+---------------+-----------+--------+----------+---------+\n"
     ]
    }
   ],
   "source": [
    "# Print out classification report\n",
    "pretty_classification_report(y_test_sparse, y_pred_lr_tfidf, \"Logistic Regression, TF-IDF\")"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e73a6816-b7f5-424e-ae44-28a43beb4024",
   "metadata": {
    "id": "fe1f5d4b-a47e-4176-8471-abded7d05efd"
   },
   "source": [
    "**Save the model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "12ac7eda-7197-4f34-9726-e2487d120b65",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-17T23:28:35.531932Z",
     "iopub.status.busy": "2023-05-17T23:28:35.531732Z",
     "iopub.status.idle": "2023-05-17T23:28:35.722656Z",
     "shell.execute_reply": "2023-05-17T23:28:35.721923Z",
     "shell.execute_reply.started": "2023-05-17T23:28:35.531914Z"
    },
    "id": "17a4761c-b1b8-47da-b197-4670f5e69252",
    "tags": []
   },
   "outputs": [],
   "source": [
    "pickle.dump(best_model_lr_tfidf, open('logistic_regression_model_tfidf.pkl', 'wb'))"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e79705d9-9bb1-417b-9849-65efc9aee540",
   "metadata": {},
   "source": [
    "# Running times\n",
    "Re-run just the optimal model to get the CPU execution time "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1c43f7fc-98e9-4fb5-b753-b3e0f54076cf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T12:18:16.698785Z",
     "iopub.status.busy": "2023-05-21T12:18:16.698565Z",
     "iopub.status.idle": "2023-05-21T12:22:19.030223Z",
     "shell.execute_reply": "2023-05-21T12:22:19.029297Z",
     "shell.execute_reply.started": "2023-05-21T12:18:16.698767Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 4min 2s, sys: 301 ms, total: 4min 2s\n",
      "Wall time: 4min 2s\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.10/site-packages/sklearn/linear_model/_sag.py:350: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(C=10, multi_class=&#x27;multinomial&#x27;, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;,\n",
       "                   tol=1e-05)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" checked><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(C=10, multi_class=&#x27;multinomial&#x27;, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;,\n",
       "                   tol=1e-05)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(C=10, multi_class='multinomial', penalty='l1', solver='saga',\n",
       "                   tol=1e-05)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Word2Vec\n",
    "lr_w2v_OPT = LogisticRegression(multi_class = 'multinomial', solver='saga', C = 10, penalty = 'l1', tol = 1e-05)\n",
    "%time lr_w2v_OPT.fit(X_train_dense, y_train_dense)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8034e768-618a-4f22-bb06-e8e836189f29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-21T12:22:19.031814Z",
     "iopub.status.busy": "2023-05-21T12:22:19.031591Z",
     "iopub.status.idle": "2023-05-21T12:25:39.788403Z",
     "shell.execute_reply": "2023-05-21T12:25:39.787494Z",
     "shell.execute_reply.started": "2023-05-21T12:22:19.031795Z"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 3min 20s, sys: 244 ms, total: 3min 20s\n",
      "Wall time: 3min 20s\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(C=1, multi_class=&#x27;multinomial&#x27;, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;,\n",
       "                   tol=0.001)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" checked><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(C=1, multi_class=&#x27;multinomial&#x27;, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;,\n",
       "                   tol=0.001)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(C=1, multi_class='multinomial', penalty='l1', solver='saga',\n",
       "                   tol=0.001)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# TF-IDF\n",
    "lr_tfidf_OPT = LogisticRegression(multi_class = 'multinomial', max_iter=100, solver='saga', C= 1, penalty= 'l1', tol= 0.001)\n",
    "%time lr_tfidf_OPT.fit(X_train_sparse, y_train_sparse)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "machine_shape": "hm",
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Python 3 ",
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
   "version": "3.10.10"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "toc-autonumbering": true,
  "toc-showcode": false,
  "toc-showmarkdowntxt": false,
  "toc-showtags": false
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
