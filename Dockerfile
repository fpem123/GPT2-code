FROM python

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget

RUN pip install --upgrade pip
RUN pip install transformers \
    torch \
    sentencepiece \
    opyrator

RUN mkdir -p /app
WORKDIR /app
COPY . .

EXPOSE 80

CMD ["opyrator", "launch-api", "my_opy:generate_crime_punish", "--port", "80"]