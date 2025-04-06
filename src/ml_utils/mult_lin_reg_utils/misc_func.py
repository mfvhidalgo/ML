from typing import List, Union, Any

def temp_remove_this_item(lst: List,item: Any):
    '''
    Takes a list, copies it, then removes a specified item
    '''
    this_list = lst.copy()
    this_list.remove(item)
    return this_list