"""This module is generally only intended for use by developers of this
package.  It provides three principle submodules for:

1) _internal.package: for examining meta data relating to this python package
2) _internal.project: for examining meta data relating to the project this package belongs to
3) _internal.log: for generating course logging information for the user
"""

def ascii_encode_value(value):
    # This will fail for strings but pass for lists, etc.
    if(hasattr(value, '__iter__')):
        result = [ascii_encode_value(value_i) for value_i in value]
    elif(hasattr(value, 'encode')):
        result = value.encode('ascii')
    else:
        result = value
    return result

def ascii_encode_dict(data):
    # Split the dictionary into keys and values
    keys = data.keys()
    values = data.values()
    # Convert keys
    keys_ascii = [key.encode('ascii') for key in keys]
    # Convert values
    values_ascii = [ascii_encode_value(value) for value in values]
    # Return dictionary as a result
    return dict(zip(keys_ascii,values_ascii))
