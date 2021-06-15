import json
from time import sleep
from json import dumps
from kafka import KafkaProducer
import requests


def getData():
  producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                           value_serializer=lambda v: json.dumps(v).encode('utf-8'))

  response = requests.get(
    "https://api.etherscan.io/api?module=account&action=txlist&address=0xaF92E5F69bd05FfF0b16A3D5135E98FB4c47c33a&startblock=0&endblock=99999999&sort=asc&apikey=A9XZBH1IY7CV7K93TB73JX525FFEEESP86"
  )

  json_data = response.json()
  data = json.dumps(json_data)
  
  producer.send('eth-transaction', json_data)


getData()

