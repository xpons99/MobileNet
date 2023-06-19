FROM public.ecr.aws/g7a8t7v6/jobs-container-keras-export-base:75ad1ee6b0d55e012ecacbac3fa7d1ce53363d55-1683534744313

WORKDIR /scripts

# Install extra dependencies here
COPY requirements.txt ./
RUN /app/keras/.venv/bin/pip3 install --no-cache-dir -r requirements.txt

# Copy all files to the home directory
COPY . ./

# The train command (we run this from the keras venv, which has all dependencies)
ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
