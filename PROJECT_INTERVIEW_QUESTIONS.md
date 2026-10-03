# Customer Support Intelligence Platform
## Interview Preparation

### Project Overview
The repository implements a customer support ticket intelligence system that predicts ticket category, priority, and estimated resolution time from ticket metadata and free-text fields. The codebase combines a Streamlit dashboard, a FastAPI model-serving layer, local ML model artifacts, and Jupyter notebooks used for training and experimentation. The actual production-facing logic is centered around `prediction_api/model_service.py`, `prediction_api/main.py`, and `app/pages/2_Prediction.py`.

### Technical Architecture
- `app/` contains the Streamlit UI and EDA dashboard.
- `app/pages/1_EDA.py` loads and visualizes ticket data from `data/sample_tickets.csv`.
- `app/pages/2_Prediction.py` collects ticket details and sends them to the API.
- `prediction_api/` contains the FastAPI application, request/response schemas, and model loader.
- `models/` contains local model artifacts for category, priority, and resolution-time prediction.
- `training/` contains notebook-based experiments for baseline ML, classical models, LSTM, BERT, priority modeling, resolution time modeling, and MLflow use.
- `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml` are present and configure the API and frontend containers.

### End-to-End Project Flow
Raw Ticket Data
↓
Data loading and standardization
↓
Feature construction (subject + description text, dates, metadata)
↓
Text preprocessing and classifier feature generation
↓
Multiple model experiments in notebooks
↓
Model artifact persistence for category, priority, and resolution time
↓
FastAPI inference service with Pydantic validation
↓
Streamlit dashboard for EDA and prediction UI
↓
Dockerized local deployment

### Section 1 — Project Understanding & Architecture

### Question 1
What is the project trying to solve, and what are the actual prediction tasks implemented in the repository?

### Answer
The repository is a customer support intelligence platform meant to help analyze support tickets and produce predictions from ticket metadata and text. The implemented tasks are reflected in the API and model-service logic: category prediction for ticket type, priority prediction, and resolution-time estimation. The README explicitly says these tasks are supported, and the code confirms they are wired into the FastAPI endpoint and model loading flow.

### Why the interviewer may ask this
They want to confirm that you understand the product goal and can map the problem to the actual implementation rather than just the generic idea of “ticket classification.”

### Key points to remember
- The system predicts category, priority, and resolution time.
- The project uses the ticket subject, description, and metadata fields.
- The actual service entry points are in `prediction_api/main.py` and `prediction_api/model_service.py`.

### Possible follow-up
How does the model service decide which predictions are available when some model artifacts are missing?

### Project reference
`README.md`, `prediction_api/main.py`, `prediction_api/model_service.py`

### Question 2
How is the repository organized architecturally, and what role does each major directory play?

### Answer
The repository is split into UI, API, data, model, training, and deployment layers. The front-end is in `app/`, the inference API is in `prediction_api/`, the dataset lives in `data/`, trained models live in `models/`, experiments are in `training/`, and deployment files are in Docker and Compose configuration at the root.

### Why the interviewer may ask this
They want to see whether you understand the separation of concerns and how a full ML application is structured beyond just notebooks.

### Key points to remember
- Streamlit handles dashboard experience.
- FastAPI handles inference requests.
- Notebooks are for training/experimentation.
- Docker config packages the app for local deployment.

### Possible follow-up
Which components are model-training code versus production-serving code?

### Project reference
`app/`, `prediction_api/`, `training/`, `docker-compose.yml`

### Question 3
What does the actual inference path look like when a user submits a ticket from the Streamlit page?

### Answer
The Streamlit prediction page collects form data and sends a JSON payload to the FastAPI service. The API validates the input using Pydantic models, loads the required models, builds a feature DataFrame, runs the category, priority, and resolution models, and returns the predictions. This path is encoded in `app/pages/2_Prediction.py` and `prediction_api/model_service.py`.

### Why the interviewer may ask this
They are checking whether you understand end-to-end request flow and the handoff between UI and backend.

### Key points to remember
- Streamlit does not load the models itself.
- The frontend calls `PREDICTION_API_URL`.
- Model inference happens in the Python service, not in the UI.

### Possible follow-up
What would happen if the API is unreachable or a model file is missing?

### Project reference
`app/pages/2_Prediction.py`, `prediction_api/main.py`, `prediction_api/model_service.py`

### Question 4
Why does the repository separate the dashboard, API, and training notebooks instead of keeping everything inside one script?

### Answer
This separation makes the project easier to maintain and more realistic for deployment. The notebooks are for experimentation and model development, while the API is for serving predictions, and the dashboard is for interactive exploration. It also allows models to be loaded locally with versioned artifacts and avoids coupling the UI to training code.

### Why the interviewer may ask this
They want to know whether you can reason about architecture trade-offs in ML projects.

### Key points to remember
- Training and serving should be separated.
- Notebook experimentation is different from runtime service code.
- Deployment is simplified by using environment variables and model paths.

### Possible follow-up
How would you scale this if many users used the app simultaneously?

### Project reference
`README.md`, `prediction_api/config.py`, `docker-compose.yml`

### Question 5
Which parts of the project are productionized and which appear to be exploratory only?

### Answer
The productionized pieces are the FastAPI service, the Streamlit interface, the Docker setup, and the model-loading logic in `prediction_api/`. The Jupyter notebooks, especially the training notebooks and MLflow exploration notebooks, are exploratory and serve as training/experiment references. The actual runtime app consumes model artifacts rather than re-training on demand.

### Why the interviewer may ask this
They want to distinguish between code that is part of an application pipeline and code that is only used for experimentation.

### Key points to remember
- `prediction_api` is the serving layer.
- `training/` is experiment-heavy.
- The dashboard reads from the API rather than directly from the training notebooks.

### Possible follow-up
Which artifacts would you expect to be versioned in a real MLOps pipeline?

### Project reference
`training/`, `prediction_api/`, `app/`

### Section 2 — Dataset & Data Preprocessing

### Question 6
What dataset is actually used in the project, and what kind of data does it contain?

### Answer
The repository uses `data/sample_tickets.csv`. From the notebooks and service code, it contains customer and ticket fields such as ticket ID, customer age, customer gender, product purchased, date of purchase, ticket subject, ticket description, ticket type, ticket priority, ticket channel, first response time, time to resolution, and customer satisfaction rating. The data appears to be a synthetic sample dataset rather than a real production ticket dataset.

### Why the interviewer may ask this
They want to see whether you know how data shape and schema drive feature design and model selection.

### Key points to remember
- The dataset is local and loaded from CSV.
- The schema includes both ticket text and non-text metadata.
- The actual code expects fields like `Ticket Subject`, `Ticket Description`, and `Date of Purchase`.

### Possible follow-up
Which fields are used for category prediction vs. resolution-time prediction?

### Project reference
`data/sample_tickets.csv`, `prediction_api/model_service.py`

### Question 7
How do the notebooks treat missing values and incomplete records in the ticket dataset?

### Answer
The training notebooks show nulls in several fields such as `Resolution`, `First Response Time`, and `Customer Satisfaction Rating`. The code uses `isnull().sum()` and then handles or drops columns depending on the modeling task. For the app and API, categorical fields are normalized with defaults and text inputs are filled with empty strings when needed.

### Why the interviewer may ask this
They want to assess whether you understand that data quality and missingness directly affect model performance.

### Key points to remember
- The notebooks explicitly inspect missing values.
- Missing values are a real issue in the dataset.
- The API supplies defaults for missing or incomplete user input.

### Possible follow-up
How would you handle missing ticket text in production?

### Project reference
`training/baselineModel.ipynb`

### Question 8
How is the ticket text constructed in this project, and why is that important?

### Answer
The project builds a combined text field from the ticket subject and description, such as `Ticket Subject + " " + Ticket Description`. This is used in the TF-IDF pipeline and in the category model input. In `ModelService`, the category model receives a concatenated string: `f"{ticket.subject} {ticket.description}"` before tokenization.

### Why the interviewer may ask this
They are testing whether you understand that text features are often formed from multiple fields and that feature engineering is not just a raw copy of a single column.

### Key points to remember
- Ticket subject and description are combined.
- Combined text is used for classification.
- The same idea is used for TF-IDF features and model inputs.

### Possible follow-up
What happens if either the subject or description is empty?

### Project reference
`prediction_api/model_service.py`, `training/baselineModel.ipynb`

### Question 9
What preprocessing steps are clearly implemented for the text columns?

### Answer
The most explicit preprocessing in the repo is text concatenation and TF-IDF vectorization with English stop-word removal. In the notebook baseline model, the pipeline uses `TfidfVectorizer(stop_words="english", max_features=5000)` on the combined text. That is the clearest text preprocessing step present in the implementation.

### Why the interviewer may ask this
They want to verify whether you can identify the feature engineering that was actually used, rather than generic NLP pipelines.

### Key points to remember
- Stop words are removed in the TF-IDF baseline.
- A `max_features` limit is used.
- Text is transformed into sparse numerical features.

### Possible follow-up
Why use a sparse representation instead of dense embedding vectors in a baseline model?

### Project reference
`training/baselineModel.ipynb`

### Question 10
What role does label encoding play in the classification workflow?

### Answer
The notebooks explicitly use `LabelEncoder` to encode the target labels for ticket type prediction. This converts string classes to numeric labels before training classification models. In the API, the model loader also accounts for `label_encoder.pkl` and uses it or falls back to the model’s `id2label` map to map class IDs back to human-readable categories.

### Why the interviewer may ask this
They want to confirm that you understand classification targets and the mapping between encoded labels and actual labels used in predictions.

### Key points to remember
- Target labels are encoded to numeric form.
- Inference converts back to actual category names.
- The API checks `LABEL_ENCODER_PATH` and `id2label` as a fallback.

### Possible follow-up
How would you handle new categories added to the dataset after model deployment?

### Project reference
`training/baselineModel.ipynb`, `prediction_api/model_service.py`

### Question 11
How were train/validation/test splits handled in the notebooks?

### Answer
The notebooks use `train_test_split` with `stratify=y` in the baseline and several experimental models. This is a standard approach to preserve class proportions in the split. The Optuna notebook also creates a hold-out test split before tuning. The repository does not show an explicit validation pipeline in the production API, but it does use train/test splitting in the training notebooks.

### Why the interviewer may ask this
They want to confirm that you know how data leakage and class balance affect model evaluation.

### Key points to remember
- `stratify=y` is used in several notebooks.
- A hold-out set is kept separate during tuning in the Optuna example.
- The API itself is not training models; it loads them.

### Possible follow-up
Why is a hold-out set important when tuning hyperparameters?

### Project reference
`training/baselineModel.ipynb`, `training/AdvacedClassicalMLModels.ipynb`

### Question 12
What kind of feature engineering is clearly present beyond raw text input?

### Answer
The project builds engineered fields such as `Days Since Purchase`, `Ticket Text`, and metadata-based inputs for the priority and resolution models. In the API, `_priority_features` computes `Days Since Purchase` from purchase date and a “first response time,” and `_resolution_features` adds `Ticket Type`, `Ticket Priority`, and textual content. These are not generic NLP features alone; they are domain-specific operational features relevant to ticket processing.

### Why the interviewer may ask this
They want to test whether you can connect feature engineering to business context and not just raw text vectors.

### Key points to remember
- Date-derived features matter in support workflows.
- Category and priority are used as features in downstream models.
- Ticket text is merged into model-specific feature frames.

### Possible follow-up
Why might `Days Since Purchase` help predict priority or resolution time?

### Project reference
`prediction_api/model_service.py`

### Section 3 — NLP & Feature Engineering

### Question 13
Why was text preprocessing required in this project?

### Answer
The project uses free-text data from ticket subject and description, and those fields contain noisy, variable-length text. Without preprocessing and vectorization, raw strings are not usable as direct model input. Standard preprocessing steps like concatenation and stop-word-aware TF-IDF convert the ticket text into a structured sparse representation suitable for classical ML classifiers.

### Why the interviewer may ask this
They want to see whether you understand why text models need a transformation from raw text to numeric features.

### Key points to remember
- Ticket text is variable length and noisy.
- Text cannot be fed directly to scikit-learn models.
- The project transforms the text into sparse feature vectors.

### Possible follow-up
Would you use the same preprocessing for all models in the repository?

### Project reference
`training/baselineModel.ipynb`, `prediction_api/model_service.py`

### Question 14
What is the actual TF-IDF setup used in the repository?

### Answer
The baseline notebook uses `TfidfVectorizer(stop_words="english", max_features=5000)` on the combined text feature. This creates a sparse TF-IDF matrix where tokens are weighted by term frequency and inverse document frequency. It is a classical NLP representation and is clearly used in the project before logistic regression and other baseline classifiers.

### Why the interviewer may ask this
They are checking whether you can describe the actual implementation and not just discuss TF-IDF in the abstract.

### Key points to remember
- The implementation is `TfidfVectorizer` from scikit-learn.
- Stop words are removed.
- A feature cap is set to 5000 terms.

### Possible follow-up
Why does a TF-IDF vectorizer produce a sparse matrix and what is the benefit?

### Project reference
`training/baselineModel.ipynb`

### Question 15
How does the project handle n-grams and vocabulary size in the text pipeline?

### Answer
The explicit code in the notebook uses a single `TfidfVectorizer` setup with `max_features=5000` and standard English stop-word filtering. It does not show a custom n-gram configuration in the visible code. That means the repository does not clearly implement a custom n-gram strategy beyond the default tokenization behavior of TF-IDF unless it appears elsewhere in the notebook beyond the snippets reviewed.

### Why the interviewer may ask this
They want to know whether you can stay disciplined and only describe what is actually in the code.

### Key points to remember
- The code clearly shows `max_features`.
- N-gram behavior is not explicitly customized in the visible pipeline.
- I would not claim explicit n-gram tuning unless it is explicitly present.

### Possible follow-up
What would you do if you wanted to include bigrams or trigrams?

### Project reference
`training/baselineModel.ipynb`

### Question 16
What is the significance of the sparse representation in this project’s TF-IDF pipeline?

### Answer
The TF-IDF output is sparse because most tickets do not contain most vocabulary terms. Sparse matrices are efficient for high-dimensional text data because they store only non-zero values. This is important for scalability and for scikit-learn classifiers trained on a large vocabulary. It is a practical choice for the baseline classification approach in this repo.

### Why the interviewer may ask this
They want to see whether you understand why text modeling often uses sparse structures in production and research.

### Key points to remember
- Sparse matrices keep memory usage manageable.
- Text vocabulary can be very large.
- This is compatible with linear classifiers and other classical models.

### Possible follow-up
Would a dense representation be preferable for a Transformer model?

### Project reference
`training/baselineModel.ipynb`

### Question 17
How is the combined ticket text used across the repository?

### Answer
The combined ticket text is used as the main text feature for the classical classification workflow and is also included in the API’s model-specific feature frames. In the notebooks, `combine_text` creates a `Combined_Text` column by concatenating `Ticket Subject` and `Ticket Description`. In the API, the service builds `Ticket Text` using those same two fields.

### Why the interviewer may ask this
They want to confirm that you understand the consistency of feature engineering across experiments and serving code.

### Key points to remember
- Subject + description is the key text signal.
- Both training and serving code reuse this pattern.
- The same concept is used for both baseline and downstream models.

### Possible follow-up
Would you ever remove the subject and rely only on the description?

### Project reference
`training/baselineModel.ipynb`, `prediction_api/model_service.py`

### Question 18
Is GloVe explicitly implemented in the repository?

### Answer
No. Based on the visible implementation, I cannot confirm GloVe embeddings are used. There are TF-IDF, LSTM, and Transformer/BERT-related notebooks, but the repository does not show a clear GloVe embedding pipeline. This is a place where the documentation may mention “embeddings” generally, but the code does not clearly verify a GloVe implementation.

### Why the interviewer may ask this
They want to test whether you can separate confirmed implementation from aspirational or generic ML claims.

### Key points to remember
- Documented but not verified in implementation.
- TF-IDF is verified.
- GloVe is not confirmed by code evidence.

### Possible follow-up
What would distinguish a verified embedding pipeline from a documented one?

### Project reference
`training/`, `README.md`

### Question 19
What is the actual deep-learning text approach visible in the repo, and is it clearly implemented as a production model?

### Answer
The repo contains notebooks named `LSTMClassification.ipynb` and `BertClassification.ipynb`, which indicates LSTM and BERT-based experimentation. The production API loads a Hugging Face Transformer model and tokenizer for the category pipeline through `AutoTokenizer` and `AutoModelForSequenceClassification`, which is a verified implementation in production code. A custom BiLSTM or custom transformer architecture is not clearly implemented in the runtime service code.

### Why the interviewer may ask this
They want to know whether you can distinguish notebook experiments from the code actually used in deployment.

### Key points to remember
- LSTM and BERT are present in notebooks.
- The API uses a Hugging Face transformer model.
- A custom BiLSTM layer is not clearly confirmed in the runtime code.

### Possible follow-up
How would you compare a classical TF-IDF model with a transformer model for this ticket task?

### Project reference
`training/LSTMClassification.ipynb`, `training/BertClassification.ipynb`, `prediction_api/model_service.py`

### Question 20
Why were different NLP approaches trialed in the project instead of using only one method?

### Answer
Different methods were explored to compare conventional sparse text representations with sequence-based and transformer-based models. The repository includes classical TF-IDF baselines and deep-learning notebooks such as LSTM and BERT. This provides a way to compare trade-offs in interpretability, compute, and modeling capability, while also aligning with the project’s multiple ticket tasks and experimentation workflow.

### Why the interviewer may ask this
They want to assess whether you understand the value of comparing approaches rather than committing to one prematurely.

### Key points to remember
- Baselines establish a reference point.
- Deep-learning approaches operate on different assumptions.
- The training notebooks are used to test and compare approaches.

### Possible follow-up
Which approach is easier to debug in production: TF-IDF or transformer models?

### Project reference
`training/`, `README.md`

### Section 4 — Machine Learning & Model Development

### Question 21
Which machine learning algorithms are clearly present in the repository?

### Answer
The repository contains a clear scikit-learn baseline using `LogisticRegression` and a TF-IDF pipeline. It also includes experiments with `MultinomialNB`, `RandomForestClassifier`, and XGBoost in notebooks. For the production category model, the service loads a Hugging Face `AutoModelForSequenceClassification` model. For priority and resolution-time models, it loads persisted scikit-learn or skops models.

### Why the interviewer may ask this
They are checking whether you can identify the actual algorithms and not just rely on the README summary.

### Key points to remember
- `LogisticRegression` is directly used.
- `MultinomialNB` and `RandomForestClassifier` appear in notebooks.
- XGBoost is used in Optuna tuning.
- Transformers are used for category prediction.

### Possible follow-up
Which of these algorithms is the most production-friendly for a fast inference API?

### Project reference
`training/baselineModel.ipynb`, `training/AdvacedClassicalMLModels.ipynb`, `prediction_api/model_service.py`

### Question 22
Why was logistic regression used as a baseline model in the notebook workflow?

### Answer
It is simple, fast, interpretable, and works well with sparse TF-IDF text representations. The notebook pipeline uses a `Pipeline` with a `TfidfVectorizer` and `LogisticRegression(max_iter=5000, random_state=42)`, which is a common baseline for multitclass text categorization. It provides a straightforward benchmark before trying more complex models.

### Why the interviewer may ask this
They want to see whether you understand why simple baselines still matter in ML workflows.

### Key points to remember
- It is easy to interpret.
- It trains quickly on sparse high-dimensional text matrices.
- It establishes a baseline performance reference.

### Possible follow-up
When would a simple model be better than a transformer model in practice?

### Project reference
`training/baselineModel.ipynb`

### Question 23
How does the repository handle the ticket category classification task in production?

### Answer
The category model is loaded from a local Transformers model directory and tokenizer directory via `AutoTokenizer` and `AutoModelForSequenceClassification`. The service concatenates subject and description, tokenizes the text with padding and truncation, runs the model in inference mode, and turns the predicted class ID back into a label using the encoder or `id2label` mapping.

### Why the interviewer may ask this
They want to confirm understanding of the actual classification pipeline and how labels are reconstructed after inference.

### Key points to remember
- It uses a sequence-classification model.
- Tokenizer and model are loaded from disk.
- Predictions are mapped to ticket categories.

### Possible follow-up
What is the role of `max_length=512` in the input pipeline?

### Project reference
`prediction_api/model_service.py`

### Question 24
How is priority prediction implemented and what inputs does it use?

### Answer
Priority prediction is implemented by loading a serialized model from `models/priority` and passing a DataFrame built by `_priority_features()`. That feature frame includes customer age, gender, product purchased, ticket type, ticket channel, satisfaction rating, days since purchase, and a combined ticket-text field. This is a classic tabular feature pipeline built around business metadata and ticket context.

### Why the interviewer may ask this
They are checking whether you understand the difference between text classification and structured prediction tasks.

### Key points to remember
- Priority uses customer and ticket metadata.
- `Days Since Purchase` is a derived feature.
- The text is included as part of the feature frame.

### Possible follow-up
Why not predict priority solely from the ticket text?

### Project reference
`prediction_api/model_service.py`

### Question 25
What is the actual resolution-time prediction task in the repo, and how is it modeled?

### Answer
The repository includes a resolution-time model loaded from `models/resolution_time/model.pkl`. The API builds a feature DataFrame that includes customer gender, customer age, product purchased, ticket type, ticket priority, ticket channel, satisfaction rating, and ticket text. The model then predicts a numeric value, which is returned as `resolution_time_hours`.

### Why the interviewer may ask this
They want to confirm your understanding of regression tasks and how they differ from classification tasks in the same product.

### Key points to remember
- This is a regression task.
- The target is a numeric estimated time.
- The output is a float in hours.

### Possible follow-up
How would you validate whether the resolution-time model is trustworthy?

### Project reference
`prediction_api/model_service.py`, `README.md`

### Question 26
What is the role of `skops` in the project, and where is it used?

### Answer
The project uses `skops` to load a persisted priority model from a `.skops` file. The code in `prediction_api/model_service.py` checks for either `priority/model.pkl` or `priority/model.skops` and loads the `.skops` variant using `skops.io.load` with a trusted module list. This is a real model-persistence strategy used by the project.

### Why the interviewer may ask this
They want to see whether you understand serialization formats and how model artifacts are consumed in deployment.

### Key points to remember
- `skops` is used for the priority model.
- The API supports either pickle or skops loading.
- This is important for portability and safe loading.

### Possible follow-up
What is the advantage of using a serialized model in inference services?

### Project reference
`prediction_api/model_service.py`, `requirements-api.txt`

### Question 27
What is the actual evidence of Optuna use in this repo?

### Answer
Optuna is clearly used in the notebook workflow, where an XGBoost tuning example creates a study and uses a search space with parameters like `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda`. It uses a TPE sampler and performs cross-validation in the objective function.

### Why the interviewer may ask this
They want to verify whether you can identify a real hyperparameter optimization workflow and explain the purpose of tuning.

### Key points to remember
- The tuning is done in an XGBoost notebook.
- A study is created with `optuna.create_study`.
- It minimizes cross-validation log loss.

### Possible follow-up
What is the difference between a validation set and a test set in a hyperparameter-tuning workflow?

### Project reference
`training/AdvacedClassicalMLModels.ipynb`

### Question 28
What are the main hyperparameters tuned in the Optuna workflow, and why are they important?

### Answer
The tuning notebook specifically searches over tree and boosting hyperparameters such as the number of estimators, maximum depth, learning rate, subsampling ratio, column subsampling, minimum child weight, gamma, and regularization terms. These parameters influence tree complexity, regularization, and how aggressively the model learns from the training data.

### Why the interviewer may ask this
They want to see whether you understand not just the tool but the model parameters that matter in practice.

### Key points to remember
- They are XGBoost hyperparameters.
- Tuning balances underfitting and overfitting.
- The objective is based on cross-validated log loss.

### Possible follow-up
How would you know whether the search space is too narrow or too broad?

### Project reference
`training/AdvacedClassicalMLModels.ipynb`

### Question 29
What is the most important difference between the project’s classical NLP baseline and the transformer-category model?

### Answer
The classical path uses TF-IDF features with a linear or tree-based model, while the category model uses a transformer architecture loaded from Hugging Face. The memory and compute patterns are different: TF-IDF is efficient and sparse, while transformer inference uses tokenization and deep contextual embeddings. The classical approach is more interpretable and lightweight, while the transformer model is more expressive but heavier.

### Why the interviewer may ask this
They want to see whether you can compare two valid approaches in the same project context.

### Key points to remember
- TF-IDF is sparse and efficient.
- Transformer models are contextual and heavier.
- The project compares them in different notebooks and the API.

### Possible follow-up
Which approach would you prefer for low-latency prediction and why?

### Project reference
`training/baselineModel.ipynb`, `prediction_api/model_service.py`

### Question 30
How would you explain the model selection strategy used across this repository?

### Answer
The repository follows a layered strategy: simple baselines are built first, more advanced classical models are tried next, and sequence/deep-learning experiments are explored in notebooks. The production API then loads the best operational model for the category task, while the priority and resolution models are persisted separately. This is a practical experimental pattern rather than a single fixed algorithm choice.

### Why the interviewer may ask this
They want to understand whether you can talk about a practical ML workflow, not just a single final model.

### Key points to remember
- Baselines come first.
- Notebooks provide model comparison.
- Serving code loads a chosen model artifact, not all candidate models.

### Possible follow-up
What would you do if the baseline consistently outperformed the transformer model on your task?

### Project reference
`training/`, `prediction_api/`

### Section 5 — Model Evaluation & Results

### Question 31
What evaluation metrics are clearly used in the repository?

### Answer
The notebooks explicitly compute and log accuracy, precision, recall, F1-score, and confusion matrices. The baseline notebook also prints `classification_report`, and the advanced notebook includes `roc_auc`, `precision`, `recall`, and `f1` in cross-validation scoring. The classification report file in `training/classification_report.txt` records per-class metrics and overall accuracy.

### Why the interviewer may ask this
They want to verify how the team evaluated classification quality and whether the evaluation choices fit the task.

### Key points to remember
- Accuracy is used.
- Precision, recall, and F1 are used.
- Confusion matrices and classification reports are present.

### Possible follow-up
What is the difference between macro and weighted F1 in a multiclass setting?

### Project reference
`training/classification_report.txt`, `training/baselineModel.ipynb`, `training/AdvacedClassicalMLModels.ipynb`

### Question 32
Is the repo showing final, production-validated model results? What should you be careful about?

### Answer
The repository contains training outputs and examples of metrics, but not a final, clean, production-validated benchmark tied to a release artifact. The classification report file shows accuracy around 0.19 for a baseline classification run, and the notebooks log many metrics, but they should not be treated as a definitive claim for the deployed service without checking the actual model artifact in the serving environment. This is important because the repo is primarily built around notebooks and local artifacts.

### Why the interviewer may ask this
They want to ensure you do not overclaim results and know the difference between exploratory output and production validation.

### Key points to remember
- Metrics are present in notebooks and text files.
- The exact final deployed performance is not clearly tied to a single artifact in the repo.
- I would avoid claiming a final accuracy value without verification.

### Possible follow-up
How would you validate the final model before declaring it ready for production?

### Project reference
`training/classification_report.txt`, `training/baselineModel.ipynb`

### Question 33
What is the significance of the confusion matrix in this project?

### Answer
The notebooks generate confusion matrices for the classification task, which help reveal which ticket classes are being confused. This is especially important in a multiclass support-ticket project, where categories may be semantically similar. The confusion matrix is also logged as an artifact in the MLflow experiment example.

### Why the interviewer may ask this
They want to confirm that you understand error analysis and class confusion, not just aggregate accuracy.

### Key points to remember
- It diagnoses misclassification patterns.
- It helps identify overlap between ticket types.
- It is part of the project’s evaluation workflow.

### Possible follow-up
If one class is repeatedly confused with another, what would you investigate?

### Project reference
`training/baselineModel.ipynb`, `training/confusion_matrix.png`

### Question 34
How were class imbalance issues addressed in the UI and training notebooks?

### Answer
The EDA dashboard explicitly visualizes class imbalance and cross-tab plots between ticket type and priority. In the training notebooks, the authors use stratified train/test splits, which is a common mitigation to preserve class distributions. The repo does not show a full custom resampling or class weighting strategy in the production API, so the imbalance is mostly observed and partly mitigated by stratification rather than a major rebalancing pipeline.

### Why the interviewer may ask this
They want to assess whether you understand that class imbalance is both a modeling and an analysis issue.

### Key points to remember
- The project notes class imbalance.
- `stratify=y` is used.
- The repo does not clearly implement advanced rebalancing methods in the main service.

### Possible follow-up
What would you do if one ticket category was extremely rare?

### Project reference
`app/pages/1_EDA.py`, `training/baselineModel.ipynb`

### Question 35
What do the model results suggest about the project’s early baselines, and how should that be interpreted?

### Answer
The baseline notebooks show low to modest classification performance, with accuracy values close to 0.20 in some runs. That indicates the task is non-trivial and that the dataset and labels may be noisy or difficult to separate using simple features. This is not a reason to claim a strong production result; it is a reason to investigate text representation, class balance, or alternative models.

### Why the interviewer may ask this
They want to see whether you can interpret underperforming baselines realistically and avoid over-optimistic conclusions.

### Key points to remember
- Some baseline performance is low.
- This is not unusual in early ticket categorization baselines.
- The project explores several alternatives to improve the result.

### Possible follow-up
What would you try next if the baseline accuracy is poor but the project is still promising?

### Project reference
`training/classification_report.txt`, `training/baselineModel.ipynb`

### Question 36
How do the notebooks compare multiple models, and why does that matter in this specific project?

### Answer
The notebooks compare logistic regression, Naive Bayes, random forests, XGBoost, LSTM, and BERT-style experiments. This matters because support-ticket classification is text-heavy and subject to label ambiguity, so comparing model families helps identify where the actual predictive signal lies. The repo does not show a single final winner in a polished final report, but it clearly shows an experimentation workflow.

### Why the interviewer may ask this
They want to know whether you appreciate a realistic model-development process rather than assuming one architecture solved everything.

### Key points to remember
- Multiple model families are explored.
- This is a pragmatic experimentation workflow.
- Different models are compared using consistent classification metrics.

### Possible follow-up
Which model family would you trust first in a production scenario, given this repo’s evidence?

### Project reference
`training/`, `README.md`

### Section 6 — Deep Learning / Transformers

### Question 37
What is the role of the transformer model in the repository, and how is it loaded?

### Answer
The transformer model is used for ticket category classification in the production API. `ModelService` loads the tokenizer and model using `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained`, then runs them with a concatenated subject/description string. This is a real Hugging Face Transformer-based inference path.

### Why the interviewer may ask this
They are testing whether you understand the exact production implementation of the model and not just the conceptual idea.

### Key points to remember
- The model is loaded from local disk.
- It is sequence classification.
- It is used in the API and not just in notebooks.

### Possible follow-up
What is the difference between a model directory and a serialized scikit-learn model in this repo?

### Project reference
`prediction_api/model_service.py`

### Question 38
How does the API use the tokenizer before calling the transformer model?

### Answer
The service builds a single text input from the ticket subject and description and passes it to the tokenizer with `return_tensors="pt"`, `truncation=True`, `padding=True`, and `max_length=512`. This converts the raw text into PyTorch tensors that the model can consume. The model then emits class logits, which are converted to probabilities using `torch.softmax`.

### Why the interviewer may ask this
They want to verify that you know the transformer inference lifecycle, including preprocessing and model output handling.

### Key points to remember
- `AutoTokenizer` converts text to token IDs.
- `padding` and `truncation` are used.
- Model logits are converted to probabilities.

### Possible follow-up
What would you do if the ticket description was much longer than 512 tokens?

### Project reference
`prediction_api/model_service.py`

### Question 39
Is the repo clearly implementing DistilBERT or another specific transformer variant beyond the general BERT workflow?

### Answer
The code does not clearly identify a DistilBERT model or a separate DistilBERT training configuration in the repository. There is a `BertClassification.ipynb` notebook and Hugging Face `AutoModelForSequenceClassification` usage, but I would not claim DistilBERT unless the model path or notebook explicitly names it. The implemented production stack is a general transformer category model, not a confirmed DistilBERT variant.

### Why the interviewer may ask this
They want to see whether you can stay precise and avoid claiming a model that is not clearly present.

### Key points to remember
- Documented but not verified in implementation.
- BERT is present as a notebook and general model family.
- DistilBERT is not confirmed in the actual code.

### Possible follow-up
What evidence would you need to claim a DistilBERT deployment?

### Project reference
`training/BertClassification.ipynb`, `prediction_api/model_service.py`

### Question 40
Why is a transformer model different from the TF-IDF baseline in terms of token processing and training paradigm?

### Answer
The TF-IDF baseline treats the text as a bag-of-words representation and learns from sparse term weights. The transformer path processes tokens through learned contextual embeddings and attention. This allows the model to incorporate word context and position, but it also requires more compute and usually more careful deployment, especially for local inference and model loading.

### Why the interviewer may ask this
They want to test whether you can compare a bag-of-words approach with contextual sequence models correctly.

### Key points to remember
- TF-IDF is a sparse lexical representation.
- Transformers use contextualized token embeddings.
- The repo contains both approaches.

### Possible follow-up
Would you prefer a transformer model for this ticket task if latency is critical?

### Project reference
`training/baselineModel.ipynb`, `prediction_api/model_service.py`

### Section 7 — MLflow, FastAPI, Streamlit & Docker

### Question 41
How is MLflow used in the repository, and what is the actual evidence?

### Answer
MLflow is used in the training notebooks rather than as part of the runtime service. The notebooks set an experiment name such as `CustomerIntelligence_Experiment`, log parameters, log metrics like accuracy, precision, recall, F1, and save classification reports and confusion-matrix artifacts. The repo does not show a production MLflow server configuration or a formal tracking URI in the app code.

### Why the interviewer may ask this
They want to know whether you understand ML experiment tracking and whether the project actually integrates it as a tool or only experiments with it in notebooks.

### Key points to remember
- MLflow appears in notebooks.
- Metrics and artifacts are logged.
- The production API does not directly use MLflow for inference.

### Possible follow-up
What would you add if you wanted MLflow to become part of a real deployment pipeline?

### Project reference
`training/baselineModel.ipynb`, `training/AdvacedClassicalMLModels.ipynb`, `training/mlflowuiviewing.ipynb`

### Question 42
What is the actual FastAPI implementation in this repo, and what endpoints are present?

### Answer
The API is defined in `prediction_api/main.py`. It exposes a health endpoint at `GET /health` and a prediction endpoint at `POST /v1/predict`. The input is validated with `TicketRequest` and the response is represented by `PredictionResponse`. It is a lightweight prediction service used by the UI and local deployment environment.

### Why the interviewer may ask this
They want to confirm your understanding of the actual HTTP surface and API design in the repo.

### Key points to remember
- Health endpoint returns readiness and model loading status.
- Prediction endpoint returns category, confidence, priority, and resolution time.
- Pydantic models validate request/response payloads.

### Possible follow-up
How would you add authentication or versioning if this became a shared service?

### Project reference
`prediction_api/main.py`, `prediction_api/schemas.py`

### Question 43
How are Pydantic models used in the project, and why are they important?

### Answer
Pydantic models define the request schema and response structure. `TicketRequest` enforces required fields like `description` and `purchase_date`, and constrains fields like customer age, confidence, and satisfaction rating. This helps ensure the API rejects malformed data before model inference begins. The response model wraps the category, priority, and estimated resolution time.

### Why the interviewer may ask this
They want to see whether you understand API contract validation and how it reduces bad inputs to the model layer.

### Key points to remember
- Validation happens before inference.
- Input contracts are explicit and typed.
- This is part of the service reliability layer.

### Possible follow-up
What would happen if `purchase_date` is missing or malformed?

### Project reference
`prediction_api/schemas.py`

### Question 44
What is the role of Streamlit in this repository, and how does it communicate with the inference layer?

### Answer
Streamlit is the user-facing dashboard. The EDA page loads the CSV and visualizes ticket distributions and text patterns. The prediction page collects customer and ticket details and sends them to `PREDICTION_API_URL` using a JSON POST request. It does not directly load models; it delegates inference to the FastAPI service. This separation mirrors a real application architecture and makes the service easier to maintain.

### Why the interviewer may ask this
They want to confirm that you understand the UI code is not the ML logic but an entry point to it.

### Key points to remember
- Streamlit is for interaction and analysis.
- Communication is through HTTP endpoints.
- It calls the API and renders results.

### Possible follow-up
What are the advantages and disadvantages of using Streamlit for a production application?

### Project reference
`app/main.py`, `app/pages/1_EDA.py`, `app/pages/2_Prediction.py`

### Question 45
How is Docker used in this repository, and what does the local deployment look like?

### Answer
The repo contains `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml`. The API container runs Uvicorn on port 8000, and the frontend container runs Streamlit on port 8501. Docker Compose wires them together and sets `PREDICTION_API_URL` for the frontend while mounting the local `./models` directory on the API container as read-only. This makes the application easy to run locally in a reproducible environment.

### Why the interviewer may ask this
They want to test whether you understand the deployment packaging and container orchestration used in the project.

### Key points to remember
- Docker is used for both API and front-end.
- The frontend depends on the API.
- Local models are mounted into the API container.

### Possible follow-up
Why is a container useful when working with ML model artifacts?

### Project reference
`Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`

### Section 8 — Troubleshooting, Challenges & Improvements

### Question 46
If the API starts but returns `503` because no models are available, what would you check first?

### Answer
I would check the configured environment variables and the actual filesystem paths in `prediction_api/config.py` and Docker Compose. The service checks for `MODEL_ROOT`, `CATEGORY_MODEL_PATH`, `LABEL_ENCODER_PATH`, `RESOLUTION_MODEL_PATH`, and `PRIORITY_MODEL_PATH`. If the model directories are missing or the path is incorrect, the service logs an error and marks the application as not ready.

### Why the interviewer may ask this
They are testing your ability to debug configuration-driven failures in production ML services.

### Key points to remember
- The API fails gracefully when models are missing.
- File paths are critical.
- The app explicitly reports load errors in `status()`.

### Possible follow-up
How would you expose a more user-friendly health response for a missing model artifact?

### Project reference
`prediction_api/config.py`, `prediction_api/model_service.py`

### Question 47
The model works in the notebook but fails when called from the FastAPI service. What would you check?

### Answer
I would check the model path, local environment variables, model artifact format, dependency versions, and whether the category model or label encoder was serialized with a different library version. In this repo, the API loads a local transformer model and various serialized models from specific paths. I would also verify the tokenization and input structure because the API expects a combined subject/description string and data frame columns in a specific schema.

### Why the interviewer may ask this
They want to see whether you understand the gap between experimentation and deployed service behavior.

### Key points to remember
- Notebook code and runtime code are not always identical.
- Namespace and path mismatches are common.
- Tensor and DataFrame format assumptions matter.

### Possible follow-up
What would you inspect first: file path, model format, or dependency version mismatch?

### Project reference
`prediction_api/model_service.py`, `prediction_api/config.py`

### Question 48
What is one of the biggest implementation risks in this project, and how would you improve it?

### Answer
One important risk is that the API depends on local model artifacts being present and correctly configured. If any model is absent or incompatible, the service may still run but return partial or missing predictions. A practical improvement would be to standardize model artifact validation, logging, and release checks so that every environment has a verified artifact set before deployment.

### Why the interviewer may ask this
They are looking for realism and operational awareness, not just model accuracy.

### Key points to remember
- The repo relies on local artifacts.
- Partial model readiness is a real concern.
- Validation should happen before deployment.

### Possible follow-up
How would you automate checks for missing or broken model files in CI?

### Project reference
`prediction_api/model_service.py`

### Question 49
What would you do if ticket categories changed or new labels were introduced after deployment?

### Answer
I would first inspect the training and label-encoding process, because the model’s class mapping is tied to the label encoder or the model configuration’s `id2label` mapping. If new classes are introduced, I would retrain or re-export the category model and the label encoder together, then verify the API still maps class IDs back to the correct labels. I would also update any dashboards or reporting logic that assumes the old categories.

### Why the interviewer may ask this
They want to know whether you understand the operational impact of label drift and model retraining.

### Key points to remember
- Class labels are part of the training contract.
- The model and encoder must stay in sync.
- New classes require retraining and validation.

### Possible follow-up
How would you detect label drift in a live ticketing system?

### Project reference
`prediction_api/model_service.py`, `training/baselineModel.ipynb`

### Question 50
What is one practical improvement you would prioritize for the current repository, and why?

### Answer
I would prioritize making the model artifact workflow more explicit and reproducible by standardizing the training outputs, storing model metadata, and validating artifacts before runtime. The repo already has notebooks, Docker, and a FastAPI service, but the actual production artifact pipeline is not fully standardized in the codebase. A clean versioning and validation flow would reduce the risk of mismatched models, stale encoders, or partial inference readiness.

### Why the interviewer may ask this
They want to assess whether you can identify operational blind spots and propose sensible next steps.

### Key points to remember
- Model artifact validation is important.
- Production readiness is more than model training.
- Reproducible deployment matters for reliability.

### Possible follow-up
If you had one extra week, what would you build first to improve production readiness?

### Project reference
`prediction_api/`, `models/`, `docker-compose.yml`

### TOP 15 QUESTIONS TO PREPARE
1. What is the project trying to solve, and what are the actual prediction tasks implemented in the repository?  
2. How is the repository organized architecturally, and what role does each major directory play?  
3. What does the actual inference path look like when a user submits a ticket from the Streamlit page?  
4. How is the ticket text constructed in this project, and why is that important?  
5. What preprocessing steps are clearly implemented for the text columns?  
6. What is the actual TF-IDF setup used in the repository?  
7. What role does label encoding play in the classification workflow?  
8. How is the category model implemented in production?  
9. How is priority prediction implemented and what inputs does it use?  
10. What is the actual resolution-time prediction task in the repo, and how is it modeled?  
11. Which machine learning algorithms are clearly present in the repository?  
12. How are train/test splits handled in the notebooks and why is that important?  
13. What evaluation metrics are clearly used in the repository?  
14. How is MLflow used in the repository, and what is the actual evidence?  
15. What is the actual FastAPI implementation in this repo, and what endpoints are present?  

### 2-MINUTE PROJECT EXPLANATION
This project is a customer support intelligence system designed to help teams understand and route support tickets more effectively. The dataset is a local ticket CSV with customer and ticket metadata, including subject, description, product, purchase date, ticket channel, priority, and resolution-related fields. In the training notebooks, the team builds text features by combining the subject and description and uses classic NLP methods such as TF-IDF with stop-word filtering, as well as experiments with logistic regression, Naive Bayes, random forest, XGBoost, LSTM, and transformer-based classification. For the production path, the API loads a Hugging Face sequence-classification model for ticket category prediction and separate serialized models for priority and resolution-time estimation. The project also includes a Streamlit dashboard for EDA and a prediction UI, and there is Docker configuration for local deployment. The repository clearly demonstrates a real end-to-end ML workflow, but the most important thing to keep honest is that the final validated production performance is not fully isolated in a single artifact or release report, and some advertised concepts such as GloVe or DistilBERT are not clearly verified in the implementation.

### END-TO-END TECHNICAL FLOW
Raw Ticket Data
↓
CSV loading and schema normalization
↓
Missing-value checks and feature engineering
↓
Subject + description text construction
↓
TF-IDF preprocessing and classical ML experiments
↓
Transformer-based category model loading/inference
↓
Priority model feature construction and prediction
↓
Resolution-time regression model prediction
↓
FastAPI validation and response formatting
↓
Streamlit dashboard and prediction page
↓
Dockerized local deployment

### TECHNOLOGY STACK
Verified in the repository:
- Python 3.11 (documented in README)
- FastAPI
- Uvicorn
- Pydantic
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Transformers / Hugging Face
- PyTorch
- MLflow (notebook-based experiment tracking)
- XGBoost
- LightGBM (declared in README and requirements)
- joblib
- skops
- Plotly
- Matplotlib
- Seaborn
- WordCloud
- Docker and Docker Compose
- Jupyter notebooks

Documented but not verified in implementation:
- GloVe embeddings
- DistilBERT specific model variant
- A final production benchmark report for the deployed model

### WEAK AREAS TO STUDY
- The production model artifact pipeline is not fully standardized in the codebase.
- The notebooks appear exploratory; some experiments are not directly wired into the service.
- MLflow usage is notebook-based rather than fully integrated into the deployed app.
- The final production performance is not clearly isolated and validated in a single reproducible artifact.
- There is no explicit evidence of a custom BiLSTM implementation in the runtime code.
- GloVe remains unverified in the current implementation.
- DistilBERT is not confirmed by code or model file naming.

### RESUME CROSS-CHECK
Supported by implementation:
- FastAPI prediction service with health and prediction endpoints.
- Streamlit dashboard and EDA page.
- Ticket classification, priority prediction, and resolution-time estimation components.
- TF-IDF classical NLP workflow.
- Transformer-based category model inference.
- Model loading and local artifact use.
- Dockerized local deployment with API + frontend.
- MLflow experiment tracking in notebooks.
- Pydantic request/response validation.

Documented but not verified in implementation:
- “Model files are loaded locally from the configured model paths.” — Verified in code.
- “LightGBM-compatible serialized models” — present in requirements and model loading logic, but the evidence is not fully tied to a final produced model in the repo.
- “GloVe embeddings” — Documented but not verified in implementation.
- “DistilBERT” — Documented but not verified in implementation.
- “Customer segmentation” notebook appears in README, but the actual production app does not clearly expose a complete customer segmentation service.

Technologies mentioned but not found as clear implementation:
- DistilBERT-specific deployment
- GloVe embedding pipeline
- Custom BiLSTM runtime serving layer
- A fully integrated MLflow tracking server in the app

Metrics that cannot be verified:
- Any final production accuracy/F1/precision/recall claim for the active deployed model is not clearly tied to a single confirmed artifact in the repo.
- Resolution-time R² or MAE values are not clearly published in the repo for the final model.

Features that appear incomplete:
- The app is useful for demo and local inference, but the model artifact pipeline looks heavier than the repo’s runtime code suggests.
- Notebooks contain many exploratory experiments, but not all are connected to a stable, final production path.
- The repository does not clearly expose a formal ML training pipeline script for repeatable retraining.

Anything to explain carefully in an interview:
- The project includes several experiments and notebooks, but the production path has a narrower, clearer runtime implementation.
- Avoid claiming “final model performance” beyond what is actually logged in the repository.
- Do not claim technologies or model families that are not clearly present in the code.

### FINAL VERIFICATION
- Exactly 50 questions were created. Yes.
- Every question has an answer. Yes.
- Every question has a “Why the interviewer may ask this” section. Yes.
- Every question has key points. Yes.
- Every question has a possible follow-up. Yes.
- Questions are based on the actual repository. Yes.
- No unsupported technologies or results were invented. Yes, with explicit caveats where needed.
- Project-specific terminology is preserved. Yes.
- The Markdown file was created in the repository. Yes.
- The file is located inside the repository root as `PROJECT_INTERVIEW_QUESTIONS.md`. Yes.

This file is ready for review and commit after a quick check of wording and numbering.

