

# names unique, [0] is default name
template_name_synonyms = {
    '0': ['NOOP'],
    '16': ['GRIP'],
    '32': ['RELEASE', "release", "pustit"],
    '64': ['point', 'ukaž', 'POINT', 'POINT_TASK'], # point towards a location
    '80': ['PICK', 'PICK_TASK', 'seber'], # pick an object
    '128': ['PLACE'], # place an object (assuming something is being held)
    '208': ['PNP'], # pick n place
    '209': ['STORE'], # pick an object an place it into a storage
    '210': ['STASH'], # pick an object an place it into the robot storage (backstage)
    '212': ['FETCH'], # take object from backstage and put it near the user
    '213': ['FETCH_TO'], # take object from backstage and put it into a storage
    '214': ['TIDY', 'TIDY_TASK', 'ukliď'], # take all objects from the front and put them into a storage
    '256': ['STOP', 'stop'],  # stop the robot arm(s)
    '?': ['put', 'polož'],

    '512': ['RETRACT'],  # retract to starting position
    '2048': ['MOVE'],  # move eef to a location

    '4294967296': ['REM_CMD_LAST', 'REMOVE_COMMAND_LAST', 'Zruš poslední příkaz', 'remove_last_command'],  # removes last added command
    '4294967297': ['REM_CMD_X', 'REMOVE_COMMAND_X', "remove_command", "Zruš příkaz"],  # removes the specified command
    '8589934592': ['DEFINE_STORAGE'], # start process of marker recognition, polyhedron calculation and addition of this entry to the ontology
    '8589934593': ['DEFINE_POSITION'], # start process of marker recognition and addition of this entry to the ontology
    '8589934594': ['BUILD_ASSEMBLY', 'BUILD', 'build', 'stavět', 'Postav', 'build_assembly'], #starts process of building a given assembly based on the recipe using aplanner
    '8589934595': ['BUILD_ASSEMBLY_CANCEL', 'BUILD_CANCEL'], #cancels process of building a given assembly based on the recipe using aplanner

    '1111111111': ['PRODUCT_REMOVE', "remove_product", "Odeber výrobek"],
}
'''
    "peg": "kolík",
    "take": "vezmi",
    "give": "podej",
    "pass": "pass",
    "glue": "nalep",
    "learn_new_task": "Nauč se novou úlohu",
    "learn_new_tower": "Nauč se novou věž",
    "demonstration": "Demonstrace",
    "define_area": "Definování oblasti",
    "define": "definuj",
    "put that": "polož to",
    "give that": "podej to",
    "tidy that": "ukliď to",
    "cube": "kostka",
    "any": "jakoukoli",
    "red":"červená",
    "red wine": "vínová",
    "dark red": "vínová",
    "magenta": "růžová",
    "blue": "modrá",
    "light blue": "světle modrá",
    "cyan": "tyrkysová",
    "green": "zelená",
    "black": "černá",
    "purple": "fialová",
    "dark purple": "tmavě fialová",
    "yellow": "žlutá",
    "gold": "zlatá",
    "white": "bílá",
    "here": "sem",
    "down": "dolů",
    "position": "pozice",
    "center": "center",
    "left": "vlevo",
    "right":"vpravo",
    "top": "nad",
    "bottom": "pod",
    "and": "a",
    "silent": "ticho",
    "talk": "mluv",
    "storage": "úložiště",
    "define_storage": "Definuj úložiště",
    "define_position": "Definuj pozici",
    "backstage": "sklad",
    "table": "stůl",
    "all": "vše",
    "cancel_build": "zruš stavění",
    "cancel":"zruš"
'''




def template_name_to_id(name):
    for key in template_name_synonyms.keys():
        if name in template_name_synonyms[key]:
            return int(key)
    return None

def template_name_to_default_name(name):
    for key in template_name_synonyms.keys():
        if name in template_name_synonyms[key]:
            return template_name_synonyms[key][0]
    return None

def tester_template_name_to_id():
    name = 'ukaz'
    assert template_name_to_id(name) == 64    
    assert template_name_to_default_name(name) == 'point'

