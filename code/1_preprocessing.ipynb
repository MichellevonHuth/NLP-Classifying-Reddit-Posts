{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b6d3e8a4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:18.290847Z",
     "start_time": "2023-05-15T18:55:18.288640Z"
    }
   },
   "outputs": [],
   "source": [
    "pip install imblearn --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ddbfdc7f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:23.598612Z",
     "start_time": "2023-05-15T18:55:18.292790Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     /Users/mathildelundsberg/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "# Import libraries and packages\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import re \n",
    "import pickle\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "\n",
    "import nltk\n",
    "from nltk import word_tokenize, pos_tag\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer, SnowballStemmer\n",
    "nltk.download('punkt')\n",
    "\n",
    "import gensim\n",
    "import gensim.downloader as api\n",
    "\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "\n",
    "np.random.seed(42)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "3dca4925",
   "metadata": {},
   "source": [
    "## Loading data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "80daa2ba",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:31.968350Z",
     "start_time": "2023-05-15T18:55:23.600910Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>selftext</th>\n",
       "      <th>subreddit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Life is so pointless without others</td>\n",
       "      <td>Does anyone else think the most important part...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Cold rage?</td>\n",
       "      <td>Hello fellow friends 😄\\n\\nI'm on the BPD spect...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I don’t know who I am</td>\n",
       "      <td>My [F20] bf [M20] told me today (after I said ...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HELP! Opinions! Advice!</td>\n",
       "      <td>Okay, I’m about to open up about many things I...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>help</td>\n",
       "      <td>[removed]</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                 title  \\\n",
       "0  Life is so pointless without others   \n",
       "1                           Cold rage?   \n",
       "2                I don’t know who I am   \n",
       "3              HELP! Opinions! Advice!   \n",
       "4                                 help   \n",
       "\n",
       "                                            selftext subreddit  \n",
       "0  Does anyone else think the most important part...       BPD  \n",
       "1  Hello fellow friends 😄\\n\\nI'm on the BPD spect...       BPD  \n",
       "2  My [F20] bf [M20] told me today (after I said ...       BPD  \n",
       "3  Okay, I’m about to open up about many things I...       BPD  \n",
       "4                                          [removed]       BPD  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import data\n",
    "raw = pd.read_csv('mental_disorders_reddit.csv')\n",
    "data = raw[['title','selftext','subreddit']] \n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a7d0403f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:33.512429Z",
     "start_time": "2023-05-15T18:55:31.969877Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1mDataframe info:\u001b[0m\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 701787 entries, 0 to 701786\n",
      "Data columns (total 3 columns):\n",
      " #   Column     Non-Null Count   Dtype \n",
      "---  ------     --------------   ----- \n",
      " 0   title      701741 non-null  object\n",
      " 1   selftext   668096 non-null  object\n",
      " 2   subreddit  701787 non-null  object\n",
      "dtypes: object(3)\n",
      "memory usage: 16.1+ MB\n",
      "None\n",
      "\n",
      "\u001b[1mUnique values:\u001b[0m\n",
      "title        621001\n",
      "selftext     563917\n",
      "subreddit         6\n",
      "dtype: int64\n",
      "\n",
      "\u001b[1mNumber of duplicate rows:\u001b[0m\n",
      "9825\n"
     ]
    }
   ],
   "source": [
    "# Key info on the data \n",
    "print(\"\\033[1mDataframe info:\\033[0m\")\n",
    "print(data.info())\n",
    "print(\"\\n\\033[1mUnique values:\\033[0m\")\n",
    "print(data.nunique())\n",
    "print(\"\\n\\033[1mNumber of duplicate rows:\\033[0m\")\n",
    "print(data.duplicated().sum())\n",
    "#print(\"\\n\\033[1mNumber of posts for each subreddit label:\\033[0m\")\n",
    "#print(data['subreddit'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ed369ddd",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:33.632232Z",
     "start_time": "2023-05-15T18:55:33.513667Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Number of posts</th>\n",
       "      <th>%</th>\n",
       "      <th>ratio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>BPD</th>\n",
       "      <td>241116</td>\n",
       "      <td>34.36</td>\n",
       "      <td>9.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Anxiety</th>\n",
       "      <td>173990</td>\n",
       "      <td>24.79</td>\n",
       "      <td>6.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>depression</th>\n",
       "      <td>156972</td>\n",
       "      <td>22.37</td>\n",
       "      <td>6.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mentalillness</th>\n",
       "      <td>53232</td>\n",
       "      <td>7.59</td>\n",
       "      <td>2.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>bipolar</th>\n",
       "      <td>51112</td>\n",
       "      <td>7.28</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>schizophrenia</th>\n",
       "      <td>25365</td>\n",
       "      <td>3.61</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Number of posts      %  ratio\n",
       "BPD                     241116  34.36    9.5\n",
       "Anxiety                 173990  24.79    6.9\n",
       "depression              156972  22.37    6.2\n",
       "mentalillness            53232   7.59    2.1\n",
       "bipolar                  51112   7.28    2.0\n",
       "schizophrenia            25365   3.61    1.0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Investigate class imbalance\n",
    "counts = data['subreddit'].value_counts()\n",
    "ratios = data['subreddit'].value_counts(normalize=True)\n",
    "pd.DataFrame({\"Number of posts\": counts, \"%\": ratios*100, \n",
    "              \"ratio\": ratios/min(data['subreddit'].value_counts(normalize=True))}).round({'%': 2, 'ratio': 1})"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "a18ab393",
   "metadata": {},
   "source": [
    "*We have a class imbalance of approximately 10:7:6:2:2:1*"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f27dd45c",
   "metadata": {},
   "source": [
    "### Dropping invalid and null rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "02541a7c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:34.102636Z",
     "start_time": "2023-05-15T18:55:33.633629Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Count null: \n",
      "title           46\n",
      "selftext     33691\n",
      "subreddit        0\n",
      "dtype: int64 \n",
      "\n",
      "Count of \"[deleted]\" in selftext: 9742\n",
      "Count of \"[removed]\" in selftext: 86875\n",
      "Count of \"[deleted]\" in title: 1\n",
      "Count of \"[removed]\" in title: 0\n"
     ]
    }
   ],
   "source": [
    "# How many null values\n",
    "print(f\"Count null: \\n{data.isna().sum()} \\n\")\n",
    "\n",
    "# How many deleted or removed posts\n",
    "print(f\"Count of \\\"[deleted]\\\" in selftext: {data['selftext'][data['selftext'] == '[deleted]'].count()}\")\n",
    "print(f\"Count of \\\"[removed]\\\" in selftext: {data['selftext'][data['selftext'] == '[removed]'].count()}\")\n",
    "print(f\"Count of \\\"[deleted]\\\" in title: {data['selftext'][data['title'] == '[deleted]'].count()}\")\n",
    "print(f\"Count of \\\"[removed]\\\" in title: {data['selftext'][data['title'] == '[removed]'].count()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "eeeed9aa",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:34.615895Z",
     "start_time": "2023-05-15T18:55:34.103989Z"
    }
   },
   "outputs": [],
   "source": [
    "# Remove posts with null values and posts that were removed or deleted \n",
    "data = data[data['selftext'] != '[removed]']\n",
    "data = data[data['selftext'] != '[deleted]']\n",
    "data = data[data['title'] != '[deleted]']\n",
    "data = data.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0f2efbc7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:34.996790Z",
     "start_time": "2023-05-15T18:55:34.617206Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Count null: 0\n",
      "Count of \"[deleted]\" and \"[removed]\" in selftext and title: 0\n"
     ]
    }
   ],
   "source": [
    "# How many null values after dropping rows\n",
    "print(f\"Count null: {data.isna().sum().sum()}\")\n",
    "\n",
    "# How many deleted or removed posts after dropping rows\n",
    "print(f\"Count of \\\"[deleted]\\\" and \\\"[removed]\\\" in selftext and title: {data['selftext'][data['selftext'] == '[deleted]'].count() + data['selftext'][data['selftext'] == '[removed]'].count() + data['selftext'][data['title'] == '[deleted]'].count() + data['selftext'][data['title'] == '[removed]'].count()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "883335c0",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:41.171179Z",
     "start_time": "2023-05-15T18:55:35.000372Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of posts containing \"[removed]\" or \"[deleted]\": 123\n"
     ]
    }
   ],
   "source": [
    "# Checking for \"[removed]\" or \"[deleted]\" in the title and selftext when the \n",
    "# title and selftext also contains other text (i.e. not the whole post deleted or removed)\n",
    "print(\"Number of posts containing \\\"[removed]\\\" or \\\"[deleted]\\\":\", \n",
    "      data['selftext'][data['selftext'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count() +\n",
    "      data['title'][data['title'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count() +\n",
    "      data['subreddit'][data['subreddit'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count())"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "e9d3a63c",
   "metadata": {},
   "source": [
    "*123 rows is not many out of 580,000 rows so we just drop these rows.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8e781f2f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:47.117274Z",
     "start_time": "2023-05-15T18:55:41.172795Z"
    }
   },
   "outputs": [],
   "source": [
    "# Dropping rows containing \"[removed]\" or \"[deleted]\"\n",
    "data = data[data['selftext'].str.contains(\"\\[deleted|deleted\\]|\\[removed|removed\\]\") == False]\n",
    "data = data[data['title'].str.contains(\"\\[deleted|deleted\\]|\\[removed|removed\\]\") == False]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f747c4c3",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:53.329752Z",
     "start_time": "2023-05-15T18:55:47.118690Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of posts containing \"[removed]\" or \"[deleted]\": 0\n"
     ]
    }
   ],
   "source": [
    "# Checking that those rows have been dropped\n",
    "print(\"Number of posts containing \\\"[removed]\\\" or \\\"[deleted]\\\":\", \n",
    "      data['selftext'][data['selftext'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count() +\n",
    "      data['title'][data['title'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count() +\n",
    "      data['subreddit'][data['subreddit'].str.contains('\\[deleted|deleted\\]|\\[removed|removed\\]')].count())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b9bd64b1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:53.551270Z",
     "start_time": "2023-05-15T18:55:53.331286Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1mDataframe info after dropping:\u001b[0m\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 571352 entries, 0 to 701786\n",
      "Data columns (total 3 columns):\n",
      " #   Column     Non-Null Count   Dtype \n",
      "---  ------     --------------   ----- \n",
      " 0   title      571352 non-null  object\n",
      " 1   selftext   571352 non-null  object\n",
      " 2   subreddit  571352 non-null  object\n",
      "dtypes: object(3)\n",
      "memory usage: 17.4+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# Key info after dropping\n",
    "print(\"\\033[1mDataframe info after dropping:\\033[0m\")\n",
    "print(data.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b574804f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:53.652604Z",
     "start_time": "2023-05-15T18:55:53.552996Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1mNumber of posts after dropping:\u001b[0m\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Number of posts</th>\n",
       "      <th>%</th>\n",
       "      <th>ratio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>BPD</th>\n",
       "      <td>205136</td>\n",
       "      <td>35.90</td>\n",
       "      <td>17.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Anxiety</th>\n",
       "      <td>160570</td>\n",
       "      <td>28.10</td>\n",
       "      <td>13.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>depression</th>\n",
       "      <td>120990</td>\n",
       "      <td>21.18</td>\n",
       "      <td>10.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mentalillness</th>\n",
       "      <td>37436</td>\n",
       "      <td>6.55</td>\n",
       "      <td>3.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>bipolar</th>\n",
       "      <td>35589</td>\n",
       "      <td>6.23</td>\n",
       "      <td>3.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>schizophrenia</th>\n",
       "      <td>11631</td>\n",
       "      <td>2.04</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Number of posts      %  ratio\n",
       "BPD                     205136  35.90   17.6\n",
       "Anxiety                 160570  28.10   13.8\n",
       "depression              120990  21.18   10.4\n",
       "mentalillness            37436   6.55    3.2\n",
       "bipolar                  35589   6.23    3.1\n",
       "schizophrenia            11631   2.04    1.0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Investigate class imbalance after dropping\n",
    "counts = data['subreddit'].value_counts()\n",
    "ratios = data['subreddit'].value_counts(normalize=True)\n",
    "print(\"\\033[1mNumber of posts after dropping:\\033[0m\")\n",
    "pd.DataFrame({\"Number of posts\": counts, \"%\": ratios*100, \n",
    "              \"ratio\": ratios/min(data['subreddit'].value_counts(normalize=True))}).round({'%': 2, 'ratio': 1})"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c013016d",
   "metadata": {},
   "source": [
    "## Format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "13285561",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:54.188121Z",
     "start_time": "2023-05-15T18:55:53.653756Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Life is so pointless without others Does anyon...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Cold rage? Hello fellow friends 😄\\n\\nI'm on th...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I don’t know who I am My [F20] bf [M20] told m...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HELP! Opinions! Advice! Okay, I’m about to ope...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>My ex got diagnosed with BPD Without going int...</td>\n",
       "      <td>BPD</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                post subreddit\n",
       "0  Life is so pointless without others Does anyon...       BPD\n",
       "1  Cold rage? Hello fellow friends 😄\\n\\nI'm on th...       BPD\n",
       "2  I don’t know who I am My [F20] bf [M20] told m...       BPD\n",
       "3  HELP! Opinions! Advice! Okay, I’m about to ope...       BPD\n",
       "5  My ex got diagnosed with BPD Without going int...       BPD"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Concatenate title and selftext columns into one feature for classification \n",
    "data[\"post\"] = data[\"title\"] + \" \" + data[\"selftext\"]\n",
    "data = data[['post', 'subreddit']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5757b5fd",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:54.233905Z",
     "start_time": "2023-05-15T18:55:54.189806Z"
    }
   },
   "outputs": [],
   "source": [
    "# Lowercase the Anxiety subreddit label (for cute team member who was annoyed by the capital letter)\n",
    "data['subreddit'][data['subreddit'] == 'Anxiety'] = 'anxiety'"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "4bcb492e",
   "metadata": {},
   "source": [
    "## Splitting & Undersampling"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f0819ebd",
   "metadata": {},
   "source": [
    "### Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "baa97d90",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:54.717987Z",
     "start_time": "2023-05-15T18:55:54.235233Z"
    }
   },
   "outputs": [],
   "source": [
    "# Splitting the data using stratified sampling (data is imbalanced) (20/80 split)\n",
    "train, test = train_test_split(data, test_size = 0.2, random_state = 42, stratify=data['subreddit'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "8b45c6e1",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:54.784643Z",
     "start_time": "2023-05-15T18:55:54.719348Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1mNumber of posts in data:\u001b[0m\n",
      "BPD              205136\n",
      "anxiety          160570\n",
      "depression       120990\n",
      "mentalillness     37436\n",
      "bipolar           35589\n",
      "schizophrenia     11631\n",
      "Name: subreddit, dtype: int64\n",
      "\n",
      "\u001b[1mNumber of posts in train:\u001b[0m\n",
      "BPD              164108\n",
      "anxiety          128456\n",
      "depression        96792\n",
      "mentalillness     29949\n",
      "bipolar           28471\n",
      "schizophrenia      9305\n",
      "Name: subreddit, dtype: int64\n",
      "\n",
      "\u001b[1mNumber of posts in test:\u001b[0m\n",
      "BPD              41028\n",
      "anxiety          32114\n",
      "depression       24198\n",
      "mentalillness     7487\n",
      "bipolar           7118\n",
      "schizophrenia     2326\n",
      "Name: subreddit, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# See stratified sampling distributions\n",
    "print(\"\\033[1mNumber of posts in data:\\033[0m\")\n",
    "print(data['subreddit'].value_counts())\n",
    "print(\"\\n\\033[1mNumber of posts in train:\\033[0m\")\n",
    "print(train['subreddit'].value_counts())\n",
    "print(\"\\n\\033[1mNumber of posts in test:\\033[0m\")\n",
    "print(test['subreddit'].value_counts())"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "fefe4a93",
   "metadata": {},
   "source": [
    "### Undersampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d439128d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.572492Z",
     "start_time": "2023-05-15T18:55:54.785795Z"
    }
   },
   "outputs": [],
   "source": [
    "# Set undersampling ratio (5:3.5:3:2:2:1) {original ratio 10:7:6:2:2:1}\n",
    "len_min_class = train['subreddit'][train['subreddit'] == 'schizophrenia'].value_counts()[0]\n",
    "rus_ratio = {'BPD': int(len_min_class*5),\n",
    "              'anxiety': int(len_min_class*3.5),\n",
    "              'depression': int(len_min_class*3),\n",
    "              'mentalillness': int(len_min_class*2),\n",
    "              'bipolar': int(len_min_class*2),\n",
    "              'schizophrenia': int(len_min_class*1)}\n",
    "\n",
    "# Instatiate under-sampler\n",
    "rus = RandomUnderSampler(sampling_strategy = rus_ratio, random_state=42)\n",
    "\n",
    "# Resample the training set only\n",
    "X_train, y_train = rus.fit_resample(train[['post']], train['subreddit'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6e0e6a10",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.585742Z",
     "start_time": "2023-05-15T18:55:56.573762Z"
    }
   },
   "outputs": [],
   "source": [
    "# Store the text data for the train set\n",
    "org_train = pd.concat([X_train, y_train], axis=1)\n",
    "\n",
    "# Test set into X and y, and store the text data for the test set\n",
    "X_test = test[['post']]\n",
    "y_test = test['subreddit']\n",
    "org_test = pd.concat([X_test, y_test], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "07f5dcd7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.620718Z",
     "start_time": "2023-05-15T18:55:56.586949Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\u001b[1mTotal number of posts/observations in train set after undersampling:\u001b[0m\n",
      "153532\n",
      "\n",
      "\u001b[1mNumber of posts in train set after undersampling:\u001b[0m\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Number of posts</th>\n",
       "      <th>%</th>\n",
       "      <th>ratio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>BPD</th>\n",
       "      <td>46525</td>\n",
       "      <td>30.30</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>anxiety</th>\n",
       "      <td>32567</td>\n",
       "      <td>21.21</td>\n",
       "      <td>3.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>depression</th>\n",
       "      <td>27915</td>\n",
       "      <td>18.18</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>bipolar</th>\n",
       "      <td>18610</td>\n",
       "      <td>12.12</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mentalillness</th>\n",
       "      <td>18610</td>\n",
       "      <td>12.12</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>schizophrenia</th>\n",
       "      <td>9305</td>\n",
       "      <td>6.06</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Number of posts      %  ratio\n",
       "BPD                      46525  30.30    5.0\n",
       "anxiety                  32567  21.21    3.5\n",
       "depression               27915  18.18    3.0\n",
       "bipolar                  18610  12.12    2.0\n",
       "mentalillness            18610  12.12    2.0\n",
       "schizophrenia             9305   6.06    1.0"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Total number of rows after undersampling\n",
    "print(\"\\n\\033[1mTotal number of posts/observations in train set after undersampling:\\033[0m\")\n",
    "print(len(y_train))\n",
    "\n",
    "# Investigate class imbalance after undersampling\n",
    "counts = y_train.value_counts()\n",
    "ratios = y_train.value_counts(normalize=True)\n",
    "print(\"\\n\\033[1mNumber of posts in train set after undersampling:\\033[0m\")\n",
    "pd.DataFrame({\"Number of posts\": counts, \"%\": ratios*100, \n",
    "              \"ratio\": ratios/min(y_train.value_counts(normalize=True))}).round({'%': 2, 'ratio': 1})"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f2292a4b",
   "metadata": {},
   "source": [
    "## Pre-processing"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "6b0a7f0a",
   "metadata": {},
   "source": [
    "### Pre-processing Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "45bfe466",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.626302Z",
     "start_time": "2023-05-15T18:55:56.622220Z"
    }
   },
   "outputs": [],
   "source": [
    "# Function to call in CountVectorizer:\n",
    "    # preprocessor = None (will happen inside tokenizer)\n",
    "    # stop_words = None (will happen inside tokenizer)\n",
    "    # tokenizer = preprocess()\n",
    "\n",
    "# Define preprocessing function for CountVectorizer \n",
    "    # (mask URLs, tokenize, remove stop words, lemmatize, stem, lowercase)\n",
    "def preprocess(text):\n",
    "    stemmer = SnowballStemmer('english')\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    preprocessed_post = []\n",
    "    \n",
    "    # Mask URLs\n",
    "    url = re.compile(r'(http?://|www\\.)\\S+')\n",
    "    text = url.sub('[url]', text)\n",
    "    \n",
    "    # Iterate through tokens and POS tags (word_tokenize also removes white spaces and line breaks)\n",
    "    for token, tag in nltk.pos_tag(gensim.utils.simple_preprocess(text)):\n",
    "        pos=tag[0].lower()\n",
    "        \n",
    "        # Set POS tag if not in list\n",
    "        if pos not in ['a', 'r', 'n', 'v']:\n",
    "            pos='n'\n",
    "        \n",
    "        # Check if token is a stop word\n",
    "        if token not in stop_words:\n",
    "            \n",
    "            # Lowercase, lemmatize and stem, then append to output list\n",
    "            preprocessed_post.append(stemmer.stem(WordNetLemmatizer().lemmatize(token.lower(), pos=pos)))\n",
    "    \n",
    "    return preprocessed_post"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2e3565eb",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.630831Z",
     "start_time": "2023-05-15T18:55:56.627664Z"
    }
   },
   "outputs": [],
   "source": [
    "# Define preprocessing function for Word2Vec \n",
    "    #(mask URLs, tokenize, lowercase, remove stopwords)\n",
    "def preprocess_w2v(text):\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    preprocessed_post = []\n",
    "    \n",
    "    # Mask URLs\n",
    "    url = re.compile(r'(http?://|www\\.)\\S+')\n",
    "    text = url.sub('[url]', text)\n",
    "    \n",
    "    # Iterate through tokens and POS tags (word_tokenize also removes white spaces and line breaks)\n",
    "    for token in gensim.utils.simple_preprocess(text):\n",
    "        \n",
    "        # Append token to output list if token is not a stop word\n",
    "        if token not in stop_words:\n",
    "            preprocessed_post.append(token)\n",
    "    \n",
    "    return preprocessed_post"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "baeecbc2",
   "metadata": {},
   "source": [
    "### Preprocess & Vectorize - CountVec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "81382445",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T18:55:56.634715Z",
     "start_time": "2023-05-15T18:55:56.632244Z"
    }
   },
   "outputs": [],
   "source": [
    "# Instatiate vectorizer and TF-IDF transformer\n",
    "vectorizer = CountVectorizer(decode_error = 'ignore', preprocessor = None, \n",
    "                             stop_words = None, tokenizer = preprocess, token_pattern=None)\n",
    "transformer = TfidfTransformer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "a7142e22",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:14:30.954973Z",
     "start_time": "2023-05-15T18:55:56.636064Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 18min 25s, sys: 8.64 s, total: 18min 34s\n",
      "Wall time: 18min 34s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<153532x56705 sparse matrix of type '<class 'numpy.float64'>'\n",
       "\twith 9137670 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "# Fit and transform train set\n",
    "X_train_sparse = vectorizer.fit_transform(X_train['post'])\n",
    "X_train_sparse = transformer.fit_transform(X_train_sparse)\n",
    "X_train_sparse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "371b79c4",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:28:38.547399Z",
     "start_time": "2023-05-15T19:14:30.956261Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<114271x56705 sparse matrix of type '<class 'numpy.float64'>'\n",
       "\twith 6877860 stored elements in Compressed Sparse Row format>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Transform test set\n",
    "X_test_sparse = vectorizer.transform(X_test['post'])\n",
    "X_test_sparse = transformer.transform(X_test_sparse)\n",
    "X_test_sparse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "7691d7b3",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:28:38.551552Z",
     "start_time": "2023-05-15T19:28:38.548639Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Verify that second dimension of train and test match\n",
    "X_train_sparse.shape[1] == X_test_sparse.shape[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "b45f812d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:28:38.562391Z",
     "start_time": "2023-05-15T19:28:38.560123Z"
    }
   },
   "outputs": [],
   "source": [
    "# Store the y values\n",
    "y_train_sparse = y_train\n",
    "y_test_sparse = y_test"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "7d95945a",
   "metadata": {},
   "source": [
    "### Preprocess & Vectorize - Word2Vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "19720b71",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:28:38.568949Z",
     "start_time": "2023-05-15T19:28:38.565070Z"
    }
   },
   "outputs": [],
   "source": [
    "# Define function to get word2vec vectors\n",
    "def get_word_vectors(tokens, model):\n",
    "    vectors = []\n",
    "    for token in tokens:\n",
    "        try:\n",
    "            vector = model[token]\n",
    "            vectors.append(vector)\n",
    "        except KeyError:\n",
    "            continue\n",
    "    return np.array(vectors)\n",
    "\n",
    "# Define function to check vector list dimensions\n",
    "# Will return dim of first elements in each level of nested list\n",
    "def dim(a):\n",
    "    if (type(a) != list) and (type(a) != np.ndarray):\n",
    "        return []\n",
    "    return [len(a)] + dim(a[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7478dd73",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:29:24.575180Z",
     "start_time": "2023-05-15T19:28:38.571630Z"
    }
   },
   "outputs": [],
   "source": [
    "# New dataframes for w2v\n",
    "w2v_train = org_train.copy()\n",
    "\n",
    "# Load pre-trained word2vec model (trained on google news dataset with ~100 billion words)\n",
    "w2v_model = api.load('word2vec-google-news-300')"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f5282f6a",
   "metadata": {},
   "source": [
    "#### Train set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3c823463",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:30:17.262836Z",
     "start_time": "2023-05-15T19:29:24.581798Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>tokenized</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Just wanted to drop a note telling you I care…...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[wanted, drop, note, telling, care, wanted, te...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Do you guys ever regret or hesitate disclosing...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[guys, ever, regret, hesitate, disclosing, bpd...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DAE feel like a hermit? I have BPD and I often...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[dae, feel, like, hermit, bpd, often, want, ho...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>my FP pushed me away feels like I'd rather bea...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[fp, pushed, away, feels, like, rather, beaten...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Feeling empowered with self hate Because I kno...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[feeling, empowered, self, hate, know, hate, a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153527</th>\n",
       "      <td>Wanted to share my plan weight loss strategy t...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[wanted, share, plan, weight, loss, strategy, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153528</th>\n",
       "      <td>Felt lonely and made a server with a few frien...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[felt, lonely, made, server, friends, talk, pe...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153529</th>\n",
       "      <td>how I figured schizophrenia out So, 10 years a...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[figured, schizophrenia, years, ago, mom, divo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153530</th>\n",
       "      <td>It's my 31st B-day tomorrow. SO far I've given...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[st, day, tomorrow, far, given, opiates, quit,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153531</th>\n",
       "      <td>Thinking I have schizophrenia made me worse Fr...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[thinking, schizophrenia, made, worse, start, ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>153532 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                     post      subreddit  \\\n",
       "0       Just wanted to drop a note telling you I care…...            BPD   \n",
       "1       Do you guys ever regret or hesitate disclosing...            BPD   \n",
       "2       DAE feel like a hermit? I have BPD and I often...            BPD   \n",
       "3       my FP pushed me away feels like I'd rather bea...            BPD   \n",
       "4       Feeling empowered with self hate Because I kno...            BPD   \n",
       "...                                                   ...            ...   \n",
       "153527  Wanted to share my plan weight loss strategy t...  schizophrenia   \n",
       "153528  Felt lonely and made a server with a few frien...  schizophrenia   \n",
       "153529  how I figured schizophrenia out So, 10 years a...  schizophrenia   \n",
       "153530  It's my 31st B-day tomorrow. SO far I've given...  schizophrenia   \n",
       "153531  Thinking I have schizophrenia made me worse Fr...  schizophrenia   \n",
       "\n",
       "                                                tokenized  \n",
       "0       [wanted, drop, note, telling, care, wanted, te...  \n",
       "1       [guys, ever, regret, hesitate, disclosing, bpd...  \n",
       "2       [dae, feel, like, hermit, bpd, often, want, ho...  \n",
       "3       [fp, pushed, away, feels, like, rather, beaten...  \n",
       "4       [feeling, empowered, self, hate, know, hate, a...  \n",
       "...                                                   ...  \n",
       "153527  [wanted, share, plan, weight, loss, strategy, ...  \n",
       "153528  [felt, lonely, made, server, friends, talk, pe...  \n",
       "153529  [figured, schizophrenia, years, ago, mom, divo...  \n",
       "153530  [st, day, tomorrow, far, given, opiates, quit,...  \n",
       "153531  [thinking, schizophrenia, made, worse, start, ...  \n",
       "\n",
       "[153532 rows x 3 columns]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Apply w2v preprocessing to each post's text and add to new column in dataframe \n",
    "w2v_train['tokenized'] = w2v_train['post'].apply(preprocess_w2v)\n",
    "w2v_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a57cf41e",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:30:17.273602Z",
     "start_time": "2023-05-15T19:30:17.264231Z"
    }
   },
   "outputs": [],
   "source": [
    "# Get tokens as a list of strings\n",
    "w2v_tokens = w2v_train['tokenized'].tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "c365ab47",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:30:52.073590Z",
     "start_time": "2023-05-15T19:30:17.275829Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 23.3 s, sys: 4.04 s, total: 27.3 s\n",
      "Wall time: 34.8 s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[153532, 56, 300]"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "# Iterate function over each post to get a list of vector arrays\n",
    "w2v_token_vectors = [get_word_vectors(toks, w2v_model) for toks in w2v_tokens]\n",
    "\n",
    "# Check dimensions - should be 3D with second dim == len(1st post) and last dimension == 300\n",
    "dim(w2v_token_vectors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "1ffe8e10",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:02.157643Z",
     "start_time": "2023-05-15T19:30:52.078027Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/mathildelundsberg/opt/anaconda3/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3474: RuntimeWarning: Mean of empty slice.\n",
      "  return _methods._mean(a, axis=axis, dtype=dtype,\n",
      "/Users/mathildelundsberg/opt/anaconda3/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  ret = ret.dtype.type(ret / rcount)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[153532, 300]"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the mean vector for each post (average of the token vectors for each post)\n",
    "w2v_post_vectors = [np.mean(token_vec, axis=0) for token_vec in w2v_token_vectors]\n",
    "    # Gives warning due to getting mean of empty vectors (∵ no words recognized by w2v, e.g. posts in chinese)\n",
    "\n",
    "# Check dimensions - should be 2D with last dimension == 300 (averaging removes a dimension)\n",
    "dim(w2v_post_vectors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b72e9296",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:02.502223Z",
     "start_time": "2023-05-15T19:32:02.165763Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>tokenized</th>\n",
       "      <th>vector</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Just wanted to drop a note telling you I care…...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[wanted, drop, note, telling, care, wanted, te...</td>\n",
       "      <td>[0.040843215, 0.049962725, 0.0064185006, 0.084...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Do you guys ever regret or hesitate disclosing...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[guys, ever, regret, hesitate, disclosing, bpd...</td>\n",
       "      <td>[0.046438448, 0.033510163, 0.043753274, 0.0814...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DAE feel like a hermit? I have BPD and I often...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[dae, feel, like, hermit, bpd, often, want, ho...</td>\n",
       "      <td>[0.070028685, 0.024488831, -0.0045175552, 0.09...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>my FP pushed me away feels like I'd rather bea...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[fp, pushed, away, feels, like, rather, beaten...</td>\n",
       "      <td>[-0.0135599775, 0.08199056, 0.07342699, 0.0509...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Feeling empowered with self hate Because I kno...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[feeling, empowered, self, hate, know, hate, a...</td>\n",
       "      <td>[0.06837972, 0.028429667, 0.059834797, 0.07486...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153527</th>\n",
       "      <td>Wanted to share my plan weight loss strategy t...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[wanted, share, plan, weight, loss, strategy, ...</td>\n",
       "      <td>[0.008502463, 0.07149074, -0.017547939, 0.1418...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153528</th>\n",
       "      <td>Felt lonely and made a server with a few frien...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[felt, lonely, made, server, friends, talk, pe...</td>\n",
       "      <td>[0.024809647, 0.02319336, -0.035194397, 0.0661...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153529</th>\n",
       "      <td>how I figured schizophrenia out So, 10 years a...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[figured, schizophrenia, years, ago, mom, divo...</td>\n",
       "      <td>[0.0016873488, 0.066772856, -0.03414416, 0.087...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153530</th>\n",
       "      <td>It's my 31st B-day tomorrow. SO far I've given...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[st, day, tomorrow, far, given, opiates, quit,...</td>\n",
       "      <td>[0.048358917, 0.016300201, 0.01368475, 0.04881...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153531</th>\n",
       "      <td>Thinking I have schizophrenia made me worse Fr...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[thinking, schizophrenia, made, worse, start, ...</td>\n",
       "      <td>[0.034528095, 0.04300944, 0.0061333976, 0.1284...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>153532 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                     post      subreddit  \\\n",
       "0       Just wanted to drop a note telling you I care…...            BPD   \n",
       "1       Do you guys ever regret or hesitate disclosing...            BPD   \n",
       "2       DAE feel like a hermit? I have BPD and I often...            BPD   \n",
       "3       my FP pushed me away feels like I'd rather bea...            BPD   \n",
       "4       Feeling empowered with self hate Because I kno...            BPD   \n",
       "...                                                   ...            ...   \n",
       "153527  Wanted to share my plan weight loss strategy t...  schizophrenia   \n",
       "153528  Felt lonely and made a server with a few frien...  schizophrenia   \n",
       "153529  how I figured schizophrenia out So, 10 years a...  schizophrenia   \n",
       "153530  It's my 31st B-day tomorrow. SO far I've given...  schizophrenia   \n",
       "153531  Thinking I have schizophrenia made me worse Fr...  schizophrenia   \n",
       "\n",
       "                                                tokenized  \\\n",
       "0       [wanted, drop, note, telling, care, wanted, te...   \n",
       "1       [guys, ever, regret, hesitate, disclosing, bpd...   \n",
       "2       [dae, feel, like, hermit, bpd, often, want, ho...   \n",
       "3       [fp, pushed, away, feels, like, rather, beaten...   \n",
       "4       [feeling, empowered, self, hate, know, hate, a...   \n",
       "...                                                   ...   \n",
       "153527  [wanted, share, plan, weight, loss, strategy, ...   \n",
       "153528  [felt, lonely, made, server, friends, talk, pe...   \n",
       "153529  [figured, schizophrenia, years, ago, mom, divo...   \n",
       "153530  [st, day, tomorrow, far, given, opiates, quit,...   \n",
       "153531  [thinking, schizophrenia, made, worse, start, ...   \n",
       "\n",
       "                                                   vector  \n",
       "0       [0.040843215, 0.049962725, 0.0064185006, 0.084...  \n",
       "1       [0.046438448, 0.033510163, 0.043753274, 0.0814...  \n",
       "2       [0.070028685, 0.024488831, -0.0045175552, 0.09...  \n",
       "3       [-0.0135599775, 0.08199056, 0.07342699, 0.0509...  \n",
       "4       [0.06837972, 0.028429667, 0.059834797, 0.07486...  \n",
       "...                                                   ...  \n",
       "153527  [0.008502463, 0.07149074, -0.017547939, 0.1418...  \n",
       "153528  [0.024809647, 0.02319336, -0.035194397, 0.0661...  \n",
       "153529  [0.0016873488, 0.066772856, -0.03414416, 0.087...  \n",
       "153530  [0.048358917, 0.016300201, 0.01368475, 0.04881...  \n",
       "153531  [0.034528095, 0.04300944, 0.0061333976, 0.1284...  \n",
       "\n",
       "[153532 rows x 4 columns]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Add dense vector embeddings to dataframe\n",
    "w2v_train['vector'] = w2v_post_vectors\n",
    "w2v_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "158a16f7",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:09.317363Z",
     "start_time": "2023-05-15T19:32:02.504412Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>tokenized</th>\n",
       "      <th>vector</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>148902</th>\n",
       "      <td>||;;:|\\\\||!; •••---•••</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>138287</th>\n",
       "      <td>7463819273636787717263669 646278-84847891‐1290...</td>\n",
       "      <td>mentalillness</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145941</th>\n",
       "      <td>ደማቸው አፍስሷል። ለድርጊቶችዎ የዘላለም ሥቃይ ይሰማዎታል። ተጠቂዎቹ ወደ...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[ደማቸው, አፍስሷል, ለድርጊቶችዎ, የዘላለም, ሥቃይ, ይሰማዎታል, ተጠቂ...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>140242</th>\n",
       "      <td>IamlivinginyourwallsIamlivinginyourwallsIamliv...</td>\n",
       "      <td>mentalillness</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122100</th>\n",
       "      <td>AAAAAAAAAAAAAAAAAAAA #AAAAAAAAAAAAAAAAAAAAAAAA...</td>\n",
       "      <td>depression</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>128690</th>\n",
       "      <td>我住在你的牆裡 我住在你的地板上 我住在你的床底下 我住在你的閣樓裡 來找我 來找我 來找我...</td>\n",
       "      <td>mentalillness</td>\n",
       "      <td>[我住在你的牆裡, 我住在你的地板上, 我住在你的床底下, 我住在你的閣樓裡, 來找我, 來...</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149186</th>\n",
       "      <td>Iི'mི iིnི yིoིuིrི wིaིlིlིsི Iི'mི iིnི yིoི...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147699</th>\n",
       "      <td>a_free_white_horse ＴＡＬＫ ＴＡＬＫ ＴＡＬＫ\\n\\nⓐⓑⓞⓤⓣ ⓘⓣ\\...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[ｔａｌｋ, ｔａｌｋ, ｔａｌｋ, neomaya]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>132644</th>\n",
       "      <td>ÄMŒGÜŠ???????? WHEN THE</td>\n",
       "      <td>mentalillness</td>\n",
       "      <td>[ämœgüš]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22908</th>\n",
       "      <td>What's pwBPD? What's pwBPD?</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[pwbpd, pwbpd]</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                     post      subreddit  \\\n",
       "148902                             ||;;:|\\\\||!; •••---•••  schizophrenia   \n",
       "138287  7463819273636787717263669 646278-84847891‐1290...  mentalillness   \n",
       "145941  ደማቸው አፍስሷል። ለድርጊቶችዎ የዘላለም ሥቃይ ይሰማዎታል። ተጠቂዎቹ ወደ...  schizophrenia   \n",
       "140242  IamlivinginyourwallsIamlivinginyourwallsIamliv...  mentalillness   \n",
       "122100  AAAAAAAAAAAAAAAAAAAA #AAAAAAAAAAAAAAAAAAAAAAAA...     depression   \n",
       "128690  我住在你的牆裡 我住在你的地板上 我住在你的床底下 我住在你的閣樓裡 來找我 來找我 來找我...  mentalillness   \n",
       "149186  Iི'mི iིnི yིoིuིrི wིaིlིlིsི Iི'mི iིnི yིoི...  schizophrenia   \n",
       "147699  a_free_white_horse ＴＡＬＫ ＴＡＬＫ ＴＡＬＫ\\n\\nⓐⓑⓞⓤⓣ ⓘⓣ\\...  schizophrenia   \n",
       "132644                            ÄMŒGÜŠ???????? WHEN THE  mentalillness   \n",
       "22908                         What's pwBPD? What's pwBPD?            BPD   \n",
       "\n",
       "                                                tokenized vector  \n",
       "148902                                                 []    NaN  \n",
       "138287                                                 []    NaN  \n",
       "145941  [ደማቸው, አፍስሷል, ለድርጊቶችዎ, የዘላለም, ሥቃይ, ይሰማዎታል, ተጠቂ...    NaN  \n",
       "140242                                                 []    NaN  \n",
       "122100                                                 []    NaN  \n",
       "128690  [我住在你的牆裡, 我住在你的地板上, 我住在你的床底下, 我住在你的閣樓裡, 來找我, 來...    NaN  \n",
       "149186                                                 []    NaN  \n",
       "147699                        [ｔａｌｋ, ｔａｌｋ, ｔａｌｋ, neomaya]    NaN  \n",
       "132644                                           [ämœgüš]    NaN  \n",
       "22908                                      [pwbpd, pwbpd]    NaN  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for the empty vectors that produced warning\n",
    "w2v_train[w2v_train['vector'].isna()].sample(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3833a053",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:09.476415Z",
     "start_time": "2023-05-15T19:32:09.319364Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "post         0\n",
       "subreddit    0\n",
       "tokenized    0\n",
       "vector       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Drop posts with empty w2v vector representations\n",
    "w2v_train = w2v_train.dropna(subset=['vector'])\n",
    "\n",
    "# Check if dropped\n",
    "w2v_train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6c3278f0",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:09.985448Z",
     "start_time": "2023-05-15T19:32:09.478256Z"
    }
   },
   "outputs": [],
   "source": [
    "# Store dense vector embeddings as numpy array\n",
    "X_train_dense = np.array(w2v_train['vector'].tolist())\n",
    "\n",
    "# Store the y values\n",
    "y_train_dense = w2v_train['subreddit']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "fa7332bf",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:32:10.011064Z",
     "start_time": "2023-05-15T19:32:09.987338Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>tokenized</th>\n",
       "      <th>vector</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Just wanted to drop a note telling you I care…...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[wanted, drop, note, telling, care, wanted, te...</td>\n",
       "      <td>[0.040843215, 0.049962725, 0.0064185006, 0.084...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Do you guys ever regret or hesitate disclosing...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[guys, ever, regret, hesitate, disclosing, bpd...</td>\n",
       "      <td>[0.046438448, 0.033510163, 0.043753274, 0.0814...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DAE feel like a hermit? I have BPD and I often...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[dae, feel, like, hermit, bpd, often, want, ho...</td>\n",
       "      <td>[0.070028685, 0.024488831, -0.0045175552, 0.09...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>my FP pushed me away feels like I'd rather bea...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[fp, pushed, away, feels, like, rather, beaten...</td>\n",
       "      <td>[-0.0135599775, 0.08199056, 0.07342699, 0.0509...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Feeling empowered with self hate Because I kno...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[feeling, empowered, self, hate, know, hate, a...</td>\n",
       "      <td>[0.06837972, 0.028429667, 0.059834797, 0.07486...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153527</th>\n",
       "      <td>Wanted to share my plan weight loss strategy t...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[wanted, share, plan, weight, loss, strategy, ...</td>\n",
       "      <td>[0.008502463, 0.07149074, -0.017547939, 0.1418...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153528</th>\n",
       "      <td>Felt lonely and made a server with a few frien...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[felt, lonely, made, server, friends, talk, pe...</td>\n",
       "      <td>[0.024809647, 0.02319336, -0.035194397, 0.0661...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153529</th>\n",
       "      <td>how I figured schizophrenia out So, 10 years a...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[figured, schizophrenia, years, ago, mom, divo...</td>\n",
       "      <td>[0.0016873488, 0.066772856, -0.03414416, 0.087...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153530</th>\n",
       "      <td>It's my 31st B-day tomorrow. SO far I've given...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[st, day, tomorrow, far, given, opiates, quit,...</td>\n",
       "      <td>[0.048358917, 0.016300201, 0.01368475, 0.04881...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>153531</th>\n",
       "      <td>Thinking I have schizophrenia made me worse Fr...</td>\n",
       "      <td>schizophrenia</td>\n",
       "      <td>[thinking, schizophrenia, made, worse, start, ...</td>\n",
       "      <td>[0.034528095, 0.04300944, 0.0061333976, 0.1284...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>153502 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                     post      subreddit  \\\n",
       "0       Just wanted to drop a note telling you I care…...            BPD   \n",
       "1       Do you guys ever regret or hesitate disclosing...            BPD   \n",
       "2       DAE feel like a hermit? I have BPD and I often...            BPD   \n",
       "3       my FP pushed me away feels like I'd rather bea...            BPD   \n",
       "4       Feeling empowered with self hate Because I kno...            BPD   \n",
       "...                                                   ...            ...   \n",
       "153527  Wanted to share my plan weight loss strategy t...  schizophrenia   \n",
       "153528  Felt lonely and made a server with a few frien...  schizophrenia   \n",
       "153529  how I figured schizophrenia out So, 10 years a...  schizophrenia   \n",
       "153530  It's my 31st B-day tomorrow. SO far I've given...  schizophrenia   \n",
       "153531  Thinking I have schizophrenia made me worse Fr...  schizophrenia   \n",
       "\n",
       "                                                tokenized  \\\n",
       "0       [wanted, drop, note, telling, care, wanted, te...   \n",
       "1       [guys, ever, regret, hesitate, disclosing, bpd...   \n",
       "2       [dae, feel, like, hermit, bpd, often, want, ho...   \n",
       "3       [fp, pushed, away, feels, like, rather, beaten...   \n",
       "4       [feeling, empowered, self, hate, know, hate, a...   \n",
       "...                                                   ...   \n",
       "153527  [wanted, share, plan, weight, loss, strategy, ...   \n",
       "153528  [felt, lonely, made, server, friends, talk, pe...   \n",
       "153529  [figured, schizophrenia, years, ago, mom, divo...   \n",
       "153530  [st, day, tomorrow, far, given, opiates, quit,...   \n",
       "153531  [thinking, schizophrenia, made, worse, start, ...   \n",
       "\n",
       "                                                   vector  \n",
       "0       [0.040843215, 0.049962725, 0.0064185006, 0.084...  \n",
       "1       [0.046438448, 0.033510163, 0.043753274, 0.0814...  \n",
       "2       [0.070028685, 0.024488831, -0.0045175552, 0.09...  \n",
       "3       [-0.0135599775, 0.08199056, 0.07342699, 0.0509...  \n",
       "4       [0.06837972, 0.028429667, 0.059834797, 0.07486...  \n",
       "...                                                   ...  \n",
       "153527  [0.008502463, 0.07149074, -0.017547939, 0.1418...  \n",
       "153528  [0.024809647, 0.02319336, -0.035194397, 0.0661...  \n",
       "153529  [0.0016873488, 0.066772856, -0.03414416, 0.087...  \n",
       "153530  [0.048358917, 0.016300201, 0.01368475, 0.04881...  \n",
       "153531  [0.034528095, 0.04300944, 0.0061333976, 0.1284...  \n",
       "\n",
       "[153502 rows x 4 columns]"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "w2v_train"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "78d52378",
   "metadata": {},
   "source": [
    "#### Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "4f33dae6",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:34:34.080203Z",
     "start_time": "2023-05-15T19:32:10.013419Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/mathildelundsberg/opt/anaconda3/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3474: RuntimeWarning: Mean of empty slice.\n",
      "  return _methods._mean(a, axis=axis, dtype=dtype,\n",
      "/Users/mathildelundsberg/opt/anaconda3/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  ret = ret.dtype.type(ret / rcount)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>post</th>\n",
       "      <th>subreddit</th>\n",
       "      <th>tokenized</th>\n",
       "      <th>vector</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5547</th>\n",
       "      <td>Lost My FP - How do I move on with my life? Ab...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[lost, fp, move, life, two, months, ago, horri...</td>\n",
       "      <td>[0.048078947, 0.049309973, 0.013398189, 0.0657...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>694157</th>\n",
       "      <td>Any tips on stopping Negative Thought Spirals?...</td>\n",
       "      <td>mentalillness</td>\n",
       "      <td>[tips, stopping, negative, thought, spirals, u...</td>\n",
       "      <td>[0.012358166, 0.083301365, 0.039498467, 0.0498...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24095</th>\n",
       "      <td>I procrastinate sleeping in my bed. Hello -- I...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[procrastinate, sleeping, bed, hello, sure, pa...</td>\n",
       "      <td>[0.03284032, 0.0360697, -0.020166585, 0.111350...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>580687</th>\n",
       "      <td>Can Anxiety Go Away Will my Anxiety go away if...</td>\n",
       "      <td>anxiety</td>\n",
       "      <td>[anxiety, go, away, anxiety, go, away, caught,...</td>\n",
       "      <td>[0.051815636, 0.06778275, -0.03916369, 0.10600...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>492521</th>\n",
       "      <td>Hot sweaty palms I have a problem where my han...</td>\n",
       "      <td>anxiety</td>\n",
       "      <td>[hot, sweaty, palms, problem, hands, get, swea...</td>\n",
       "      <td>[0.055758916, 0.03955738, -0.022751266, 0.0956...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>344317</th>\n",
       "      <td>I wish I had a personality around others I hav...</td>\n",
       "      <td>depression</td>\n",
       "      <td>[wish, personality, around, others, things, fi...</td>\n",
       "      <td>[0.061346635, 0.042177945, 0.016557112, 0.1414...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496047</th>\n",
       "      <td>I've been feeling like I'm going to have an an...</td>\n",
       "      <td>anxiety</td>\n",
       "      <td>[feeling, like, going, anxiety, attack, second...</td>\n",
       "      <td>[0.10407967, 0.0719615, -0.06305537, 0.0552810...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62633</th>\n",
       "      <td>How do I handle my friend with BPD? We are fri...</td>\n",
       "      <td>BPD</td>\n",
       "      <td>[handle, friend, bpd, friends, nearly, ten, ye...</td>\n",
       "      <td>[0.037718367, 0.059844818, -0.0048405863, 0.09...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>341060</th>\n",
       "      <td>I’m about to check in on my ex I’m so ready to...</td>\n",
       "      <td>depression</td>\n",
       "      <td>[check, ex, ready, bury, tonight, seen, ex, fo...</td>\n",
       "      <td>[0.021391585, 0.056926005, -0.008232529, 0.081...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>608351</th>\n",
       "      <td>Anxiety about eating I don't know why, I've al...</td>\n",
       "      <td>anxiety</td>\n",
       "      <td>[anxiety, eating, know, always, good, appetite...</td>\n",
       "      <td>[0.038361646, 0.08143353, -0.026668154, 0.1532...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>114256 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                     post      subreddit  \\\n",
       "5547    Lost My FP - How do I move on with my life? Ab...            BPD   \n",
       "694157  Any tips on stopping Negative Thought Spirals?...  mentalillness   \n",
       "24095   I procrastinate sleeping in my bed. Hello -- I...            BPD   \n",
       "580687  Can Anxiety Go Away Will my Anxiety go away if...        anxiety   \n",
       "492521  Hot sweaty palms I have a problem where my han...        anxiety   \n",
       "...                                                   ...            ...   \n",
       "344317  I wish I had a personality around others I hav...     depression   \n",
       "496047  I've been feeling like I'm going to have an an...        anxiety   \n",
       "62633   How do I handle my friend with BPD? We are fri...            BPD   \n",
       "341060  I’m about to check in on my ex I’m so ready to...     depression   \n",
       "608351  Anxiety about eating I don't know why, I've al...        anxiety   \n",
       "\n",
       "                                                tokenized  \\\n",
       "5547    [lost, fp, move, life, two, months, ago, horri...   \n",
       "694157  [tips, stopping, negative, thought, spirals, u...   \n",
       "24095   [procrastinate, sleeping, bed, hello, sure, pa...   \n",
       "580687  [anxiety, go, away, anxiety, go, away, caught,...   \n",
       "492521  [hot, sweaty, palms, problem, hands, get, swea...   \n",
       "...                                                   ...   \n",
       "344317  [wish, personality, around, others, things, fi...   \n",
       "496047  [feeling, like, going, anxiety, attack, second...   \n",
       "62633   [handle, friend, bpd, friends, nearly, ten, ye...   \n",
       "341060  [check, ex, ready, bury, tonight, seen, ex, fo...   \n",
       "608351  [anxiety, eating, know, always, good, appetite...   \n",
       "\n",
       "                                                   vector  \n",
       "5547    [0.048078947, 0.049309973, 0.013398189, 0.0657...  \n",
       "694157  [0.012358166, 0.083301365, 0.039498467, 0.0498...  \n",
       "24095   [0.03284032, 0.0360697, -0.020166585, 0.111350...  \n",
       "580687  [0.051815636, 0.06778275, -0.03916369, 0.10600...  \n",
       "492521  [0.055758916, 0.03955738, -0.022751266, 0.0956...  \n",
       "...                                                   ...  \n",
       "344317  [0.061346635, 0.042177945, 0.016557112, 0.1414...  \n",
       "496047  [0.10407967, 0.0719615, -0.06305537, 0.0552810...  \n",
       "62633   [0.037718367, 0.059844818, -0.0048405863, 0.09...  \n",
       "341060  [0.021391585, 0.056926005, -0.008232529, 0.081...  \n",
       "608351  [0.038361646, 0.08143353, -0.026668154, 0.1532...  \n",
       "\n",
       "[114256 rows x 4 columns]"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Repeat dense w2v embedding for test set\n",
    "w2v_test = org_test.copy()\n",
    "w2v_test['tokenized'] = w2v_test['post'].apply(preprocess_w2v)\n",
    "\n",
    "w2v_tokens = w2v_test['tokenized'].tolist() #overwrites list for train set\n",
    "w2v_token_vectors = [get_word_vectors(toks, w2v_model) for toks in w2v_tokens] #overwrites list for train set\n",
    "w2v_post_vectors = [np.mean(token_vec, axis=0) for token_vec in w2v_token_vectors] #overwrites list for train set\n",
    "\n",
    "w2v_test['vector'] = w2v_post_vectors\n",
    "w2v_test = w2v_test.dropna(subset=['vector'])\n",
    "print(w2v_train.isna().sum().sum())\n",
    "\n",
    "X_test_dense = np.array(w2v_test['vector'].tolist())\n",
    "y_test_dense = w2v_test['subreddit']\n",
    "\n",
    "w2v_test"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f25b8ad0",
   "metadata": {},
   "source": [
    "## Pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "1676a14d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:34:34.101569Z",
     "start_time": "2023-05-15T19:34:34.087509Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "type(X_train_sparse) = <class 'scipy.sparse.csr.csr_matrix'>; X_train_sparse.shape = (153532, 56705)\n",
      "type(y_train_sparse) = <class 'pandas.core.series.Series'>; y_train_sparse.shape = (153532,)\n",
      "\n",
      "type(X_test_sparse) = <class 'scipy.sparse.csr.csr_matrix'>; X_test_sparse.shape = (114271, 56705)\n",
      "type(y_test_sparse) = <class 'pandas.core.series.Series'>; y_test_sparse.shape = (114271,)\n",
      "\n",
      "type(org_train) = <class 'pandas.core.frame.DataFrame'>; org_train.shape = (153532, 2)\n",
      "type(org_test) = <class 'pandas.core.frame.DataFrame'>; org_test.shape = (114271, 2)\n"
     ]
    }
   ],
   "source": [
    "# Check all X and y and splits are stored correctly\n",
    "print(f\"{type(X_train_sparse) = }; {X_train_sparse.shape = }\")\n",
    "print(f\"{type(y_train_sparse) = }; {y_train_sparse.shape = }\")\n",
    "print()\n",
    "print(f\"{type(X_test_sparse) = }; {X_test_sparse.shape = }\")\n",
    "print(f\"{type(y_test_sparse) = }; {y_test_sparse.shape = }\")\n",
    "print()\n",
    "print(f\"{type(org_train) = }; {org_train.shape = }\")\n",
    "print(f\"{type(org_test) = }; {org_test.shape = }\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "f6bbbc8d",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:34:34.107185Z",
     "start_time": "2023-05-15T19:34:34.103228Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "type(X_train_dense) = <class 'numpy.ndarray'>; X_train_dense.shape = (153502, 300)\n",
      "type(y_train_dense) = <class 'pandas.core.series.Series'>; y_train_dense.shape = (153502,)\n",
      "\n",
      "type(X_test_dense) = <class 'numpy.ndarray'>; X_test_dense.shape = (114256, 300)\n",
      "type(y_test_dense) = <class 'pandas.core.series.Series'>; y_test_dense.shape = (114256,)\n",
      "\n",
      "type(w2v_train) = <class 'pandas.core.frame.DataFrame'>; w2v_train.shape = (153502, 4)\n",
      "type(w2v_test) = <class 'pandas.core.frame.DataFrame'>; w2v_test.shape = (114256, 4)\n"
     ]
    }
   ],
   "source": [
    "print(f\"{type(X_train_dense) = }; {X_train_dense.shape = }\")\n",
    "print(f\"{type(y_train_dense) = }; {y_train_dense.shape = }\")\n",
    "print()\n",
    "print(f\"{type(X_test_dense) = }; {X_test_dense.shape = }\")\n",
    "print(f\"{type(y_test_dense) = }; {y_test_dense.shape = }\")\n",
    "print()\n",
    "print(f\"{type(w2v_train) = }; {w2v_train.shape = }\")\n",
    "print(f\"{type(w2v_test) = }; {w2v_test.shape = }\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f290aed3",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:34:37.789310Z",
     "start_time": "2023-05-15T19:34:34.108818Z"
    }
   },
   "outputs": [],
   "source": [
    "# Pickle the sparse vectors and y vals\n",
    "pickle.dump((X_train_sparse, X_test_sparse, y_train_sparse,\n",
    "            y_test_sparse, org_train, org_test), open('pickles/sparse.pkl', 'wb'))\n",
    "\n",
    "# Pickle the vectorizer and transformer\n",
    "pickle.dump((vectorizer, transformer), open('pickles/vectorizer.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "4a2f83cd",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:35:00.431044Z",
     "start_time": "2023-05-15T19:34:37.791839Z"
    }
   },
   "outputs": [],
   "source": [
    "# Pickle the dense vectors and y vals\n",
    "pickle.dump((X_train_dense, X_test_dense, y_train_dense,\n",
    "            y_test_dense, w2v_train, w2v_test), open('pickles/dense.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "146866b0",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:35:00.850262Z",
     "start_time": "2023-05-15T19:35:00.434594Z"
    }
   },
   "outputs": [],
   "source": [
    "# Load sparse\n",
    "X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse, org_train, org_test = pd.read_pickle(\"pickles/sparse.pkl\")\n",
    "\n",
    "# Load the vectorizer and transformer\n",
    "vectorizer, transformer = pd.read_pickle(\"pickles/vectorizer.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ff4e2cea",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-05-15T19:35:19.977872Z",
     "start_time": "2023-05-15T19:35:00.852946Z"
    }
   },
   "outputs": [],
   "source": [
    "# Load dense\n",
    "X_train_dense, X_test_dense, y_train_dense, y_test_dense, w2v_train, w2v_test = pd.read_pickle(\"pickles/dense.pkl\")"
   ]
  }
 ],
 "metadata": {
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
  "notify_time": "30",
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "calc(100% - 180px)",
    "left": "10px",
    "top": "150px",
    "width": "302.391px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
