#!/bin/bash

set -e  

for test in bin/*; do
    ./$test
done
