
# Only EN
# names unique,
# Note:
#   dict keys are ids
#   first item of list is the ret of: is_default_name()


template_name_synonyms = {
    '-1': ['noop'],
    '0': ['stop'],  # stop the robot arm(s)
    
    '1': ['pick', 'PICK_TASK', 'PickTask', 'pick up', 'lift', 'use', 'pick-up'], #'seber'], # pick an object
    '2': ['point', 'POINT_TASK', 'PointTask'], #'ukaz', 'ukaž'],
    '3': ['pass', 'pass me', 'give me', 'bring', 'need', 'pass-me'],
    '4': ['release', 'place', 'put down'], # place an object (assuming something is being held)
    '5': ['push'],
    '6': ['put-into', 'put', 'PutTask', 'put into'], #, 'polož'],
    
    '11': ['move-up', 'move up', 'up', 'move away', 'move up'], # 'nahoru'],
    '12': ['move-down', 'move down', 'down', 'move closer', 'move table'],
    '13': ['move-left', 'move left', 'left'],
    '14': ['move-right', 'move right', 'right'],
    
    '20': ['pour', 'pour into'], # 'nalij', 'nalit'],
    '21': ['stack'],
    
    '30': ['replace', 'swap', 'change'],
    
    '40': ['unglue'],
    '41': ['close', 'grip'],
    '42': ['open', "pustit"],
    
    #'208': ['pnp'], # pick n place
    #'209': ['store'], # pick an object an place it into a storage
    #'210': ['stash'], # pick an object an place it into the robot storage (backstage)
    #'212': ['fetch'], # take object from backstage and put it near the user
    #'213': ['fetch_to'], # take object from backstage and put it into a storage
    #'214': ['TIDY', 'TIDY_TASK', 'ukliď'], # take all objects from the front and put them into a storage
    
    
    #'512': ['retract'],  # retract to starting position
    #'2048': ['move'],  # move eef to a location

    #'4294967296': ['rem_cmd_last', 'REMOVE_COMMAND_LAST', 'Zruš poslední příkaz', 'remove_last_command'],  # removes last added command
    #'4294967297': ['rem_cmd_x', 'REMOVE_COMMAND_X', "remove_command", "Zruš příkaz"],  # removes the specified command
    #'8589934592': ['define_storage'], # start process of marker recognition, polyhedron calculation and addition of this entry to the ontology
    #'8589934593': ['define_position'], # start process of marker recognition and addition of this entry to the ontology
    #'8589934594': ['build_assembly', 'BUILD', 'build', 'stavět', 'Postav', 'build_assembly'], #starts process of building a given assembly based on the recipe using aplanner
    #'8589934595': ['build_assembly_cancel', 'BUILD_CANCEL'], #cancels process of building a given assembly based on the recipe using aplanner

    #'1111111111': ['product_remove', "remove_product", "Odeber výrobek"],

}

selections_name_synonyms = {
    '0': ['box'],
    '1': ['alt box'],
    '2': ['table'],
    '3': ['aruco box'],

    '10': ['cube'],
    '11': ['peg'],
    '12': ['wrench'],
    '13': ['paper'],
    '14': ['cup'],
    '16': ['can'],
    '17': ['crackers'],
    '18': ['bowl'],
    '19': ['drawer_socket', 'drawer_cabinet', 'drawer cabinet', "drawer socket"],
    '21': ['cleaner'],
    '22': ['foam'],
    '23': ['crackers'],


    # YCB
    '100': ['tomato soup can'], 
    '101': ['potted meat can'],
    '102': ['bowl'],
    '103': ['cup'],
    # Crow
    '200': ['cube holes'],
    '201': ['wheel'],
    '202': ['sphere'],

    '1000': ['glued wrench'],
    
}

storages_name_synonyms = {
    '24': ['box'],
    '0': ['paper box'],
    '1': ['abstract marked zone'],
    '2': ['out of table'],
    '3': ['on the table'],
    '4': ['bowl'],
    '5': ['drawer_socket', 'drawer_cabinet', 'drawer cabinet', "drawer socket"],
    '6': ['crackers'],
    '7': ['foam'],
    '8': ['cup'],
    '16': ['can'],
    '17': ['crackers'],
    '18': ['bowl'],
    '19': ['drawer_socket', 'drawer_cabinet', 'drawer cabinet', "drawer socket"],
    '21': ['cleaner'],
    '22': ['foam'],
    '23': ['cube'],
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
    assert isinstance(name, str), f"name is not string, it is {type(name)}"
    for key in template_name_synonyms.keys():
        if name in template_name_synonyms[key]:
            return key
    raise Exception(f"Exception for {name}")

def to_default_name(name, ct='template'):
    name = name.lower()
    if '_od_' in name:
        name = name.split("_od_")[0]
    name = name.replace("_", " ")
    assert isinstance(name, str), f"name is not string, it is {type(name)}"
    ct_name_synonyms = eval(ct+'_name_synonyms')

    for key in ct_name_synonyms.keys():
        for item in ct_name_synonyms[key]:
            if name == item.lower():
                return ct_name_synonyms[key][0]
    raise Exception(f"Exception for {name} not in {ct_name_synonyms}")


class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'


def make_conjunction(gesture_templates, language_templates, gesture_likelihoods, language_likelihoods, ct='template', keep_only_items_in_c_templates=False, c_templates=None):
    ''' template or objects or storages is names template here!
        If language and gesture templates has different sizes or one/few templates are missing
        This function makes UNION from both template lists.
    '''
    
    assert len(gesture_templates) == len(gesture_likelihoods), "items & likelihoods different sizes"
    assert len(language_templates) == len(language_likelihoods), "items & likelihoods different sizes"
    assert len(gesture_templates) == 0 or isinstance(gesture_templates[0], str), "names must be string"
    assert len(language_templates) == 0 or isinstance(language_templates[0], str), f"names must be string {language_templates}"
    assert len(gesture_likelihoods) == 0 or isinstance(gesture_likelihoods[0], float), "likelihoods must be float"
    assert len(language_likelihoods) == 0 or isinstance(language_likelihoods[0], float), "likelihoods must be float"

    #print(f"[conj fun][{len(gesture_templates)}] gesture_templates: {gesture_templates}") 
    #print(f"[conj fun][{len(language_templates)}] language_templates: {language_templates}")

    for i in range(len(gesture_templates)):
        gesture_templates[i] = to_default_name(gesture_templates[i], ct=ct)
    for i in range(len(language_templates)):
        language_templates[i] = to_default_name(language_templates[i], ct=ct)
    

    gesture_templates = list(gesture_templates)
    language_templates = list(language_templates)
    gesture_likelihoods = list(gesture_likelihoods)
    language_likelihoods = list(language_likelihoods)

    if keep_only_items_in_c_templates:
        assert c_templates is not None
        
        # print(f"c_{ct}", c_templates)        
        discards_g = []
        for n,gt in enumerate(gesture_templates):
            if gt not in c_templates:
                print(f"WARNING! {gt} was discarded!")
                discards_g.append(n)
        discards_g.reverse()
        # print("discards_g", discards_g)
        # print(f"gesture_{ct}", gesture_templates)
        for i in discards_g:
            gesture_templates.pop(i)
            gesture_likelihoods.pop(i)
        
        discards_l = []        
        # print("discards_l", discards_l)
        # print(f"language_{ct}", language_templates)

        for n,lt in enumerate(language_templates):
            if lt not in c_templates:
                print(f"WARNING! {lt} was discarded!")
                discards_l.append(n)
        discards_l.reverse()
        for i in discards_l:
            language_templates.pop(i)
            language_likelihoods.pop(i)




    extended_list = gesture_templates.copy()
    extended_list.extend(language_templates)
    unique_list = list(set(extended_list))
    
    gesture_likelihoods_unified =  [0.] * len(unique_list)
    language_likelihoods_unified = [0.] * len(unique_list)
    for unique_item in unique_list:
        if unique_item in gesture_templates:
            n = gesture_templates.index(unique_item)
            
            m = unique_list.index(unique_item)
            gesture_likelihoods_unified[m] = gesture_likelihoods[n]

        if unique_item in language_templates:
            n = language_templates.index(unique_item)
            m = unique_list.index(unique_item)
            language_likelihoods_unified[m] = language_likelihoods[n]
    
    if keep_only_items_in_c_templates:
        for tmpl in c_templates:
            if tmpl not in unique_list:
                # print(f"WARNING! {tmpl} was added with 0.0 prob!")
                unique_list.append(tmpl)
                language_likelihoods_unified.append(0.0)
                gesture_likelihoods_unified.append(0.0)
    
    #print(f"[conj fun][{len(unique_list)}] final templates: {unique_list}")
    return unique_list, language_likelihoods_unified, gesture_likelihoods_unified




if __name__ == '__main__':
    vv =make_conjunction(gesture_templates=['point', 'push', 'move up', 'stop'], language_templates=['point', 'move up', 'stop', 'push'], gesture_likelihoods=[0.1,0.2,0.3,0.4], language_likelihoods=[0.5,0.6,0.7,0.8])
    print(vv)