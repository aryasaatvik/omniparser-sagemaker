#!/bin/bash
set -e

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Script configuration
REPOSITORY_NAME="omniparser"
REGION=${AWS_DEFAULT_REGION:-$(aws configure get region)}
REGION=${REGION:-us-west-2}
SKIP_WEIGHTS=${SKIP_WEIGHTS:-false}

# Require model bucket if not skipping weights
if [ "$SKIP_WEIGHTS" = false ] && [ -z "${OMNIPARSER_MODEL_BUCKET}" ]; then
    echo "ERROR: OMNIPARSER_MODEL_BUCKET environment variable must be set (or set SKIP_WEIGHTS=true)"
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install it first."
        exit 1
    fi

    # Check requirements.txt exists
    if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
        log_error "requirements.txt not found in project root"
        exit 1
    fi
}

get_account_id() {
    log_info "Getting AWS account ID..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [ $? -ne 0 ]; then
        log_error "Failed to get AWS account ID. Please check your AWS credentials."
        exit 1
    fi
    return 0
}

create_repository() {
    log_info "Checking if ECR repository exists..."
    if ! aws ecr describe-repositories --repository-names "${REPOSITORY_NAME}" > /dev/null 2>&1; then
        log_info "Creating ECR repository ${REPOSITORY_NAME}..."
        aws ecr create-repository --repository-name "${REPOSITORY_NAME}" > /dev/null
        if [ $? -ne 0 ]; then
            log_error "Failed to create ECR repository."
            exit 1
        fi
    else
        log_info "Repository already exists."
    fi
}

create_model_bucket() {
    if [ "$SKIP_WEIGHTS" = true ]; then
        log_info "Skipping model bucket creation (SKIP_WEIGHTS=true)"
        return 0
    fi

    log_info "Checking if S3 bucket exists: ${OMNIPARSER_MODEL_BUCKET}"
    if ! aws s3api head-bucket --bucket "${OMNIPARSER_MODEL_BUCKET}" 2>/dev/null; then
        log_info "Creating S3 bucket ${OMNIPARSER_MODEL_BUCKET}..."
        if [[ "${REGION}" == "us-east-1" ]]; then
            aws s3api create-bucket --bucket "${OMNIPARSER_MODEL_BUCKET}" > /dev/null
        else
            aws s3api create-bucket --bucket "${OMNIPARSER_MODEL_BUCKET}" \
                --create-bucket-configuration LocationConstraint="${REGION}" > /dev/null
        fi
        if [ $? -ne 0 ]; then
            log_error "Failed to create S3 bucket."
            exit 1
        fi
    else
        log_info "Bucket already exists."
    fi
}

login_to_ecr() {
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    if [ $? -ne 0 ]; then
        log_error "Failed to login to ECR."
        exit 1
    fi
}

prepare_and_upload_weights() {
    if [ "$SKIP_WEIGHTS" = true ]; then
        log_info "Skipping model weights preparation (SKIP_WEIGHTS=true)"
        return 0
    fi

    log_info "Preparing and uploading model weights..."
    
    # Create a temporary directory for model weights
    TEMP_DIR=$(mktemp -d)
    log_info "Created temporary directory: ${TEMP_DIR}"
    
    # Download weights from Hugging Face
    log_info "Downloading weights from Hugging Face..."
    cd "${PROJECT_ROOT}/model"
    python3 download_weights.py --target-dir "${TEMP_DIR}"
    if [ $? -ne 0 ]; then
        log_error "Failed to download model weights."
        rm -rf "${TEMP_DIR}"
        exit 1
    fi
    
    # Create tarball
    log_info "Creating model weights tarball..."
    cd "${TEMP_DIR}"
    tar -czf model.tar.gz icon_detect icon_caption_florence
    if [ $? -ne 0 ]; then
        log_error "Failed to create model weights tarball."
        rm -rf "${TEMP_DIR}"
        exit 1
    fi
    
    # Upload to S3
    log_info "Uploading model weights to S3..."
    aws s3 cp model.tar.gz "s3://${OMNIPARSER_MODEL_BUCKET}/model/omniparser-v2/model.tar.gz"
    if [ $? -ne 0 ]; then
        log_error "Failed to upload model weights to S3."
        rm -rf "${TEMP_DIR}"
        exit 1
    fi
    
    # Cleanup
    cd - > /dev/null
    rm -rf "${TEMP_DIR}"
    log_info "Successfully uploaded model weights to S3"
}

build_image() {
    log_info "Building Docker image..."
    cd "${PROJECT_ROOT}"
    
    docker image build --platform=linux/amd64 --provenance=false --output type=docker,oci-mediatypes=false \
        -t ${REPOSITORY_NAME} \
        -f container/Dockerfile .

    if [ $? -ne 0 ]; then
        log_error "Failed to build Docker image."
        exit 1
    fi
}

tag_and_push_image() {
    local fullname="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:latest"
    
    log_info "Tagging image as ${fullname}..."
    docker tag ${REPOSITORY_NAME} ${fullname}
    if [ $? -ne 0 ]; then
        log_error "Failed to tag Docker image."
        exit 1
    fi
    
    log_info "Pushing image to ECR..."
    docker push ${fullname}
    if [ $? -ne 0 ]; then
        log_error "Failed to push Docker image to ECR."
        exit 1
    fi
}

cleanup() {
    log_info "Cleaning up..."
    # Add cleanup tasks if needed
}

main() {
    log_info "Starting build and push process..."
    if [ "$SKIP_WEIGHTS" = true ]; then
        log_warn "Model weights preparation is DISABLED (SKIP_WEIGHTS=true)"
    fi
    
    # Check requirements
    check_requirements
    
    # Get AWS account ID
    get_account_id
    
    # Create ECR repository if it doesn't exist
    create_repository
    
    # Create S3 bucket if it doesn't exist
    create_model_bucket
    
    # Login to ECR
    login_to_ecr
    
    # Prepare and upload model weights
    prepare_and_upload_weights
    
    # Build Docker image
    build_image
    
    # Tag and push image
    tag_and_push_image
    
    # Cleanup
    cleanup
    
    log_info "Successfully built and pushed image to ECR!"
    log_info "Image URI: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:latest"
    if [ "$SKIP_WEIGHTS" = false ]; then
        log_info "Model weights stored in: s3://${OMNIPARSER_MODEL_BUCKET}/model/omniparser-v2/model.tar.gz"
    fi
}

# Run main function
main 