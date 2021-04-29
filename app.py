from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template
import torch

from queue import Queue, Empty
from threading import Thread
import time

app = Flask(__name__)

tokenizer_java = AutoTokenizer.from_pretrained('microsoft/CodeGPT-small-java')
model_java = AutoModelForCausalLM.from_pretrained('microsoft/CodeGPT-small-java')

tokenizer_py = AutoTokenizer.from_pretrained('microsoft/CodeGPT-small-py')
model_py = AutoModelForCausalLM.from_pretrained('microsoft/CodeGPT-small-py')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_java.to(device)
model_py.to(device)

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1


##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    code_type = requests['input'].pop(0)
                    if code_type == 'python':
                        requests["output"] = mk_python_code(requests['input'][0], requests['input'][1], requests['input'][2])
                    elif code_type == 'java':
                        requests["output"] = mk_java_code(requests['input'][0], requests['input'][1], requests['input'][2])
                except:
                    requests["output"] = 'error :('

handler = Thread(target=handle_requests_by_batch).start()


##
# GPT-2 generator.
# Make java code!.
def mk_java_code(text, length, how_many):
    try:
        input_ids = tokenizer_java.encode(text, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        length = length if length > 0 else 1

        # model generating
        sample_outputs = model_java.generate(input_ids, pad_token_id=50256,
                                             do_sample=True,
                                             max_length=length,
                                             min_length=min_length,
                                             top_p=0.80,
                                             top_k=40,
                                             num_return_sequences=how_many)

        result = dict()

        for idx, sample_output in enumerate(sample_outputs):
            java_code = tokenizer_java.decode(sample_output, skip_special_tokens=True)
            result[idx] = java_code

        return result

    except Exception as e:
        print('Error occur in java code generating!', e)
        return jsonify({'error': e}), 500


##
# GPT-2 generator.
# Make java code!.
def mk_python_code(text, length, how_many):
    try:
        input_ids = tokenizer_py.encode(text, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        length = length if length > 0 else 1

        # model generating
        sample_outputs = model_py.generate(input_ids, pad_token_id=50256,
                                           do_sample=True,
                                           max_length=length,
                                           min_length=min_length,
                                           top_p=0.80,
                                           top_k=40,
                                           num_return_sequences=how_many)

        result = dict()

        for idx, sample_output in enumerate(sample_outputs):
            python_code = tokenizer_py.decode(sample_output, skip_special_tokens=True)
            result[idx] = python_code

        return result

    except Exception as e:
        print('Error occur in python code generating!', e)
        return jsonify({'error': e}), 500


##
# Get post request page.
@app.route('/<types>', methods=['POST'])
def generate(types):
    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        args = []

        text = request.form['text']
        length = int(request.form['length'])
        how_many = int(request.form['howmany'])


        args.append(types)
        args.append(text)
        args.append(length)
        args.append(how_many)

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


##
# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200


##
# Sever health checking page.
@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


##
# Main page.
@app.route('/')
def main():
    return render_template('main.html'), 200


if __name__ == '__main__':
    from waitress import serve
    #app.run(host='0.0.0.0', port=80)
    serve(app, host='0.0.0.0', port=80)
