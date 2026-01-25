# Faster Whisper STT Deployment Guide

This guide covers deploying Faster Whisper STT to replace AWS Transcribe.

## Architecture Overview

```
Call Audio → Kinesis Video Stream → Lambda (kinesis_stt_lambda.py)
                                         ↓
                                    Faster Whisper STT Server (ws://44.216.12.223:8766)
                                         ↓
                                    SQS FIFO Queue (TranscriptPlaybackQueue.fifo)
                                         ↓
                                    ARI Bridge (sqs_reciever.js)
```

## Components

### 1. Faster Whisper STT Server (Already Running)
- **Location**: GPU Server (44.216.12.223)
- **Port**: 8766
- **Service**: `faster-whisper-stt.service`
- **Status**: Running alongside TTS on port 8765

### 2. Kinesis STT Lambda (NEW)
- **File**: `kinesis_stt_lambda.py`
- **Purpose**: Receives audio from Kinesis, sends to STT server, forwards transcripts to SQS

## Deployment Steps

### Step 1: Package the Lambda

```bash
# On your local machine
cd sync2-tts-service/faster-whisper-stt

# Create a deployment package
mkdir -p lambda_package
pip install -r lambda_requirements.txt -t lambda_package/
cp kinesis_stt_lambda.py lambda_package/

# Create ZIP
cd lambda_package
zip -r ../faster-whisper-stt-lambda.zip .
cd ..
```

### Step 2: Create Lambda Function (AWS Console or CLI)

```bash
# Create the Lambda function
aws lambda create-function \
  --function-name faster-whisper-stt-consumer \
  --runtime python3.11 \
  --handler kinesis_stt_lambda.lambda_handler \
  --zip-file fileb://faster-whisper-stt-lambda.zip \
  --role arn:aws:iam::723307322955:role/LambdaKinesisRole \
  --timeout 60 \
  --memory-size 512 \
  --environment "Variables={
    STT_SERVER_URL=ws://44.216.12.223:8766,
    TRANSCRIPT_QUEUE_URL_GROUP=https://sqs.us-east-1.amazonaws.com/723307322955/TranscriptPlaybackQueue.fifo,
    PGHOST=pharmasync-prod-db.c85cg0m4gghf.us-east-1.rds.amazonaws.com,
    PGPORT=5432,
    PGDATABASE=pharmasync,
    PGUSER=postgres,
    PGPASSWORD=gill5477dbc,
    AWS_REGION=us-east-1
  }"
```

### Step 3: Configure Kinesis Trigger

**Option A: Replace cleanTranscriptLambda trigger**
- Remove Kinesis trigger from `cleanTranscriptLambda`
- Add Kinesis trigger to `faster-whisper-stt-consumer`

**Option B: Use environment variable toggle (Recommended)**
Keep both Lambdas with the same Kinesis trigger, and use `USE_FASTER_WHISPER_STT` env var to control which one processes:

1. Add `USE_FASTER_WHISPER_STT=true` to `faster-whisper-stt-consumer`
2. Add `USE_FASTER_WHISPER_STT=false` to `cleanTranscriptLambda`
3. Add early return in each Lambda based on the env var

### Step 4: Update ARI Bridge .env

```bash
# In Sync2ARIend/.env
USE_FASTER_WHISPER_STT=true
FASTER_WHISPER_STT_URL=ws://44.216.12.223:8766
```

### Step 5: Verify Deployment

```bash
# Check STT server is running
curl http://44.216.12.223:8766/health

# Check Lambda logs
aws logs tail /aws/lambda/faster-whisper-stt-consumer --follow

# Make a test call and verify transcripts appear in SQS
```

## Switching Between AWS Transcribe and Faster Whisper

### To use Faster Whisper STT:
1. Enable the Lambda trigger on `faster-whisper-stt-consumer`
2. Disable the Lambda trigger on `cleanTranscriptLambda`
3. Set `USE_FASTER_WHISPER_STT=true` in ARI bridge .env

### To use AWS Transcribe (rollback):
1. Enable the Lambda trigger on `cleanTranscriptLambda`
2. Disable the Lambda trigger on `faster-whisper-stt-consumer`
3. Set `USE_FASTER_WHISPER_STT=false` in ARI bridge .env

## Monitoring

### STT Server Logs
```bash
ssh ec2-user@44.216.12.223
sudo journalctl -u faster-whisper-stt -f
```

### Lambda Logs
```bash
aws logs tail /aws/lambda/faster-whisper-stt-consumer --follow
```

### SQS Queue Metrics
- Monitor `ApproximateNumberOfMessagesVisible`
- Check `NumberOfMessagesSent` to verify transcripts are flowing

## Troubleshooting

### No transcripts appearing in SQS
1. Check Lambda CloudWatch logs for errors
2. Verify STT server is accessible from Lambda VPC
3. Check SQS queue URL is correct

### High latency
1. Increase Lambda memory (more CPU)
2. Check STT server GPU utilization
3. Verify network latency between Lambda and STT server

### Barge-in not working
1. Check VAD confidence thresholds in server.py
2. Verify barge_in events are being sent to SQS
3. Check ARI bridge is processing barge_in message type

## Service URLs

| Service | URL | Port |
|---------|-----|------|
| TTS (Qwen3) | ws://44.216.12.223:8765/tts/stream | 8765 |
| STT (Faster Whisper) | ws://44.216.12.223:8766 | 8766 |
| SQS Queue | TranscriptPlaybackQueue.fifo | - |
| ARI Bridge | ec2-54-83-97-181.compute-1.amazonaws.com | 8088 |
