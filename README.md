# OPYRATOR TEST

opyrator를 테스트 해보기 위한 레파지토리입니다. 

Google's public model google/reformer-crime-and-punishment  사용해 볼 수 있습니다. 

Model: [Huggingface](https://huggingface.co/google/reformer-crime-and-punishment)


## How to run

    opyrator launch-ui my_opy:generate_crime_punish [--port port number]

    or

    opyrator launch-api my_opy:generate_crime_punish [--port port number]


## Try it!

### Post parameter

    {
      "text": "string",
      "length": int [1 ~ 200],
      "how_many": int [1 ~ 5],
      "top_k": int [1 ~ 100],
      "top_p": float [0.1 ~ 1.0],
      "do_sample": Boolean 
    }

### Output format

    {
        "message": {
          "0": "generated text, type string",
          "1": "generated text, type string",
          ...
        }
    }

<br>

### With CLI

#### Input example

    curl -X 'POST' \
      'https://master-opyrator-test-fpem123.endpoint.ainize.ai/call' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "text": "That is the",
      "length": 30,
      "how_many": 3,
      "top_k": 50,
      "top_p": 0.8,
      "do_sample": true
    }'

#### Output example

    {
        "message": {
            "0": "That is the fate. For a moage copecks, all looked at him intently at him, but",
            "1": "That is the comfort, but there is no one I have suffering. Fortenance, for the re",
            "2": "That is the honourable. There is no reformed, setting with anything he was sunant"   
        }
    }


## With API

API page: [Ainize](https://ainize.ai/fpem123/opyrator-test?branch=master)

## With Demo

Demo page: [Endpoint](https://master-opyrator-test-fpem123.endpoint.ainize.ai/)

<br>
<hr>
