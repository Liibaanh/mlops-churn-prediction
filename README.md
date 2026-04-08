# MLOps Churn Prediction API

This project demonstrates an end-to-end Machine Learning pipeline, from experimentation with MLflow to containerization with Docker and deployment to Amazon AWS ECR.

## 🚀 Project Overview
The goal of this project is to predict customer churn using a machine learning model. It is designed with MLOps best practices in mind, ensuring that the environment is reproducible and the model is ready for cloud deployment.

## 🛠 Tech Stack
- **Python**: Core programming language.
- **Scikit-learn**: For model development.
- **MLflow**: Used for experiment tracking and model logging.
- **Docker**: For containerizing the application.
- **AWS ECR**: Amazon Elastic Container Registry for hosting the Docker image.
- **GitHub Actions**: (Optional/if applicable) For CI/CD workflows.

## 📦 Containerization & Cloud
The application is fully containerized. You can find the public Docker image on 
[AWS ECR Public Gallery](https://gallery.ecr.aws/c1y4d5x4?page=1).

### To run the project locally using Docker:
1. Build the image:
   ```bash
   docker build -t churn-prediction .