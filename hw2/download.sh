#!/bin/bash

mkdir ./model

gdown 10i442TZT3obhS-9vyB6dXS6_SOHwSdc_ -O ./model/mc.zip
gdown 12EwWUqm-G0CqvafyJiS5F4xcUTBsXmrN -O ./model/qa.zip

unzip ./model/mc.zip -d ./model
unzip ./model/qa.zip -d ./model