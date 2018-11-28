from django.shortcuts import render
import json
import pandas as pd

import json
# Create your views here.
def index(request):
    df = pd.read_json('.data/ElABC_2000.json', orient='columns')
    return render(request, 'ficheros.html')

