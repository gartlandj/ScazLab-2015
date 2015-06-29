import csv
import json
import os.path
import cgitb

cgitb.enable()


class TaskHTN:

  def __init__(self, task_name=None, order=None, action_set=[]):
    self._children = []
    self._values = {'name': task_name, 'actions': action_set, 'order_invariant': order, 'support_actions': []}

  def add_child(self,childHTN):
    self._children.append(childHTN)

  def remove_child(self,child):
    if child in self._children:
      self._children.remove(child)
      return True
    else:
      return False

  @property
  def children(self):
    return self._children

  @property
  def values(self):
    return self._values

  def is_order_invariant(self):
    return self._values['order_invariant'] is True

  def is_order_agnostic(self):
    return not self.is_order_invariant()

  def to_json(self):
    return json.dumps(self.to_dict())

  def to_dict(self):
    self_dict = {'values': self._values, 'children': []}

    if len(self._children) > 0:
      self_dict['children'] = [c.to_dict() for c in self._children]    

    return self_dict

  def get_path_to_node(target_node, path=[]):
    for c in self._children:
      next_step = get_path(c, path)
      if next_step is not None: 
        path.extend(next_step)
        break

    if target_node == self: 
      return [self]
    elif length(path) > 0:
      return path

    return None

  def save(self, filename, overwrite=False):
    if os.path.isfile(filename):
      if overwrite is False:
        return False
      print "Overwriting existing file..."
    f = open(filename, 'w')
    my_json = self.to_json()
    f.write(my_json)
    f.close()


  def load(filename):
    if os.path.isfile(filename) is False:
      return None

    htn_json = None
    with open(filename,'r') as f:
      htn_json = f.read()

    return TaskHTN.from_json(htn_json)

  def from_json(json):
    root = TaskHTN(None)
    root._values = json['values']
    for c in json['children']:
      root.add_child(TaskHTN.from_json(c))
    return root

class State:

  def __init__(self, feature_vector):
    self._features = feature_vector




def num_of_spaces(string):
	spaces = 0
	for characters in string:
		if bool(characters.isspace()) == True:
			spaces += 1
		else:
			break
	
	return spaces

def retrieve_text(row):
	global rows
	print "rows"
	print rows
	r = rows[row].strip()
	print "rows[row]"
	print rows[row]
	print "r"
	print r
	if r[0] == '*' or r[0] == '-':
		r = r.replace('*','')
		r = r.replace('-','')
	return r

def ordering_class(row):
	global rows
	prefix = rows[row].strip()
	if prefix[0] == '*':
		prefix = "clique"
		return prefix
	elif prefix[0] == '-':
		prefix = "chain"
		return prefix
	else:
		prefix = "primitive"
		return prefix

def pop_stack_sib(spaces_in_current_row):
	global num_spaces_per_row
	global stack
	for i in range(1, len(num_spaces_per_row)):
		print num_spaces_per_row, spaces_in_current_row, i
		if spaces_in_current_row >= num_spaces_per_row[-i]:
			stack.pop()
			num_spaces_per_row.pop()
			print "popped"
			break


def pop_stack_parent(spaces_in_current_row):
	global num_spaces_per_row
	global stack
	for i in range(1, len(num_spaces_per_row)):
		print num_spaces_per_row, spaces_in_current_row, i
		if spaces_in_current_row == num_spaces_per_row[-i]:
			to_pop = i
			for x in range(to_pop):
			#	print stack[-1]
			#	print "look"
				stack.pop()
				num_spaces_per_row.pop()
				print "popped"
			break




def run():
	crs = open("HTN_text.txt", "r")
	rows = crs.readlines()
	row_at = 0


	num_rows = len(rows)
	num_spaces_per_row = []
	stack = [None]
	indent_leval = None 
	p = None
	while(True):


		if row_at < num_rows:
			row = rows[row_at]
			last_row = row_at - 1
			prev_row = rows[last_row]
			clean_row = retrieve_text(row_at)
			order = ordering_class(row_at)
			print order
			new_node = TaskHTN(clean_row, order)
			p = stack[-1]
			

			if row_at > 0:
				

				if p == None:
					p = new_node
					indent_leval = 0
				
				elif num_of_spaces(prev_row) < num_of_spaces(row): #child
					p.children.append(new_node)
				
				elif num_of_spaces(prev_row) == num_of_spaces(row): #sibling
					pop_stack_sib(num_of_spaces(row))
					p = stack[-1]
					

					if p == None:
						p = new_node
					else:
						p.children.append(new_node)
					
				elif num_of_spaces(prev_row) > num_of_spaces(row): #parent
					pop_stack_parent(num_of_spaces(row))
					p = stack[-1]
					

					if p == None:
						p = new_node
					else:
						p.children.append(new_node)
			

			else:
				
				if p == None:
					p = new_node
		

		else:
			print '\n'
			#print stack[1].to_json()
			json_array = stack[1].to_json()
			print "Content-Type: application/json"
			print "\n"
			print "\n"
			print stack[1].to_json()
			#print json.dumps(json_array, indent=1)
			print "\n"
			#return json.dumps(json_array)
			print '\n'

			break
		stack.append(new_node)
		num_spaces_per_row.append(num_of_spaces(row))
		row_at += 1
		print "\n\n"