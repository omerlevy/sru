import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from torch.autograd import Variable


class Contextualizer(nn.Module):
    
    def __init__(self, dims, dropout, rnn_dropout, bidirectional, architecture, tanh_input):
        super(Contextualizer, self).__init__()
        self.dims = dims
        self.dropout = nn.Dropout(dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.bidirectional = bidirectional
        self.architecture = architecture
        self.tanh_input = tanh_input
        if bidirectional:
            raise Exception('Not implemented yet')
            #self.cells = [(Cell(dims[layer], dims[layer+1], architecture).cuda(), Cell(dims[layer], dims[layer+1], architecture).cuda()) for layer in range(len(dims) - 1)]
        else:
            self.cells = nn.ModuleList([Cell(dims[layer], dims[layer+1], architecture).cuda() for layer in range(len(dims) - 1)])
        
    def forward(self, inputs, states):
        inputs = inputs.cuda()
        if self.tanh_input:
            inputs = F.tanh(inputs)
        if self.bidirectional:
            raise Exception('Not implemented yet')
        else:
            new_states = []
            for cell, state in zip(self.cells, states):
                outputs, state = self.iterate(cell, inputs, state)
                new_states.append(state)
                inputs = outputs
            return outputs, new_states
    
    def iterate(self, cell, inputs, state):
        batch_size = inputs.size()[1]
        inputs_mask = self.rnn_dropout(Variable(torch.ones([batch_size, cell.input_size]).cuda()))
        hidden_state_mask = self.rnn_dropout(Variable(torch.ones([batch_size, cell.hidden_size]).cuda())) 
        outputs = []
        inputs = [x.squeeze(0) for x in inputs.split(split_size=1, dim=0)]
        for x in inputs:
            output_mask = self.dropout(Variable(torch.ones([batch_size, cell.hidden_size]).cuda()))
            output, state = cell(x, state, inputs_mask, hidden_state_mask, output_mask)
            outputs.append(output)
        return torch.stack(outputs, 0), state
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return [(self.zeros(batch_size, dim), self.zeros(batch_size, dim)) for dim in self.dims[1:]]
        else:
            return [self.zeros(batch_size, dim) for dim in self.dims[1:]]
    
    def zeros(self, batch_size, dim):
        if self.architecture.dual_state():
            return Variable(torch.zeros(batch_size, dim).cuda()), Variable(torch.zeros(batch_size, dim).cuda())
        else:
            return Variable(torch.zeros(batch_size, dim).cuda())


class Cell(nn.Module):

    def __init__(self, input_size, hidden_size, architecture):
        super(Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.architecture = architecture
        
        # Content
        content = self.architecture.content
        if content.has_transformation:
            if content.has_state:
                self.w_content = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size).cuda())
            else:
                self.w_content = nn.Parameter(torch.Tensor(input_size, hidden_size).cuda())
        if content.has_bias:
            self.b_content = nn.Parameter(torch.Tensor(hidden_size).cuda())
        
        # Gates
        gates = self.architecture.gates    
        if gates.has_transformation:
            gate_input_dims = 0
            if gates.is_state_arg:
                gate_input_dims += hidden_size
            if gates.is_content_arg:
                gate_input_dims += hidden_size
            if gates.is_input_arg:
                gate_input_dims += input_size
            self.w_gates = nn.Parameter(torch.Tensor(gate_input_dims, gates.num_gates() * hidden_size).cuda())
        self.b_gates = nn.Parameter(torch.Tensor(gates.num_gates() * hidden_size).cuda())
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                nn.init.xavier_uniform(p.data)
            else:
                p.data.zero_()

    def forward(self, inputs, state, inputs_mask, hidden_state_mask, output_mask):
        # State
        if self.architecture.dual_state():
            hidden_state, cell_state = state
        else:
            hidden_state = cell_state = state
        
        # RNN Dropout
        dropped_inputs = inputs_mask * inputs
        dropped_hidden_state = hidden_state_mask * hidden_state
        
        # Content
        if self.architecture.content.has_transformation:
            if self.architecture.content.has_state:
                content_args = torch.cat([dropped_inputs, dropped_hidden_state], -1)
            else:
                content_args = dropped_inputs
            content = torch.mm(content_args, self.w_content)
        else:
            content = inputs # TODO Should this be dropped out? Technically, the dropout is for the matrices.
        if self.architecture.content.has_bias:
            content = content + self.b_content
        if self.architecture.content.has_tanh:
            content = F.tanh(content)
        
        # Gates - Computation
        if self.architecture.gates.has_transformation:
            args = []
            if self.architecture.gates.is_state_arg:
                args.append(dropped_hidden_state)
            if self.architecture.gates.is_content_arg:
                args.append(content) # TODO Should this be dropped out?
            if self.architecture.gates.is_input_arg:
                args.append(dropped_inputs)
            gates = torch.mm(torch.cat(args, -1), self.w_gates) + self.b_gates
        else:
            gates = self.b_gates
        
        # Gates - Aggregation
        num_gates = self.architecture.gates.num_gates()
        # Softmax
        if self.architecture.gates.is_softmax:
            gates = torch.view(torch.view(gates, [-1, self.hidden_size, num_gates]), [-1, num_gates])
            gates = F.softmax(gates)
            gates = torch.view(gates, [-1, self.hidden_size, num_gates])
            gates = [torch.unsqueeze(gate, -1) for gate in torch.split(gates, 1, -1)]
            
            new_cell_state = gates[0] * cell_state + gates[1] * content
            if self.architecture.gates.has_highway:
                new_cell_state += gates[2] * inputs
            output = new_hidden_state = new_cell_state
            output = output_mask * output # TODO This is different because it includes the highway
        # Sigmoid
        else:
            gates = torch.split(F.sigmoid(gates), self.hidden_size, -1)
            if self.architecture.gates.is_coupled:
                new_cell_state = gates[0] * cell_state + (1 - gates[0]) * content
                gates = gates[1:]
            else:
                new_cell_state = gates[0] * cell_state + gates[1] * content
                gates = gates[2:]
            new_hidden_state = new_cell_state
            if self.architecture.gates.has_tanh:
                new_hidden_state = F.tanh(new_hidden_state)
            if self.architecture.gates.has_zero_gate:
                new_hidden_state = gates[0] * new_hidden_state
            output = new_hidden_state
            output = output_mask * output # TODO This is different because it includes the highway
            if self.architecture.gates.has_highway:
                output = gates[-1] * output + (1 - gates[-1]) * inputs
        
        return output, (new_hidden_state, new_cell_state)


class Architecture:
    
    def __init__(self, content_str, gates_str):
        self.content = Content(content_str)
        self.gates = Gates(gates_str)
    
    def dual_state(self):
        return (not self.gates.is_softmax) and (self.gates.has_tanh or self.gates.has_zero_gate)
        

class Content:
    
    def __init__(self, content_str):
        if re.match(r'(tanh)?W?xh?b?', content_str).group(0) != content_str:
            raise Exception('Bad content architecture.')
        self.has_transformation = 'W' in content_str
        self.has_state = 'xh' in content_str
        self.has_bias = 'b' in content_str
        self.has_tanh = 'tanh' in content_str


class Gates:
    
    def __init__(self, gates_str):
        #TODO there is also an architecture where each argument independently gates itself; i.e. the args of the state gate are only the state, the args of the content gates are only the content, etc.
        if re.match(r's?c?i?b-((sig)|(max))-(tanh)?((\(sc\))|(sc))z?i?', gates_str).group(0) != gates_str:
            raise Exception('Bad gate architecture.')
        args, aggregation, gates = gates_str.split('-')
        # Gate Arguments
        self.is_state_arg = 's' in args
        self.is_content_arg = 'c' in args
        self.is_input_arg = 'i' in args
        self.has_transformation = args != 'b'
        # Aggregation Method
        self.is_softmax = aggregation == 'max'
        # Gating Mechanism
        if self.is_softmax and (('tanh' in gates) or ('(sc)' in gates)):
            raise Exception('Bad gate architecture.')
        self.has_tanh = 'tanh' in gates
        self.is_coupled = '(sc)' in gates
        self.has_zero_gate = 'z' in gates
        self.has_highway = 'i' in gates
    
    def num_gates(self):
        if self.is_coupled:
            num = 1
        else:
            num = 2
        if self.has_zero_gate:
            num += 1
        if self.has_highway:
            num += 1
        return num


