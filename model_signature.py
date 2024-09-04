# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:47:00 2024

@author: ayhan
"""

import json

signature_str=str(signature)
aa=json.dumps(signature_str)
signature_dict = json.loads(aa)
print("model signature is set")
input_schema = Schema([ColSpec(**col) for col in signature_dict['inputs']])
output_schema = Schema([ColSpec(**col) for col in signature_dict['outputs']])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
print("signature created succesfully")


aa=signature.to_dict()
aaa=str(aa).replace("\'", "\"")

aaa = aaa.replace("\'", "\"")
final_dictionary = json.loads(aaa)