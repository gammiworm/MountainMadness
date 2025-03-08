FROM continuumio/miniconda3

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml

ARG ENV_NAME=ai-env

# Set environment variables for Conda
ENV CONDA_DEFAULT_ENV=$ENV_NAME
ENV PATH="/opt/conda/envs/$ENV_NAME/bin:$PATH"

# Ensure Conda is using the correct environment
RUN echo "source activate $ENV_NAME" > ~/.bashrc
SHELL ["/bin/bash", "-c"]

COPY . .
CMD ["python", "app.py"]  # No need for "conda run", as the correct path is already set
