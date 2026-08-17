// Enterprise self-managed equivalent of Cloud Build (Vertex AI CI/CD).
//
// Documentation only: this repo's CI runs on GitHub Actions (see .github/workflows/ci.yml).
// There is no Jenkins server in this local/no-cloud technical test, so this pipeline
// is not executed here. It would run on a Jenkins agent that has `uv` and `docker`
// available, reusing the SAME make targets as GitHub Actions — the Makefile is the
// single source of truth for the CI steps (DRY).
pipeline {
    agent any

    stages {
        stage('Setup')    { steps { sh 'uv sync' } }
        stage('Lint')     { steps { sh 'make lint' } }
        stage('Format')   { steps { sh 'make format-check' } }
        stage('Test')     { steps { sh 'make test' } }
        stage('Build')    { steps { sh 'make docker-build' } }
        stage('Pipeline') { steps { sh 'make pipeline' } }
    }
}
