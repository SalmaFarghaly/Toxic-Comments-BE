# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020

@author: win10
"""
from pydantic import BaseModel
from typing import List
# 2. Class which describes and validates the data sent in the request (it is expecting array of strings which are the tweets)
class comment(BaseModel):
    tweets: List[str] 
