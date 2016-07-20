local Rack, parent = torch.class('nn.Rack', 'nn.Module')

function Rack:__init(alpha, inputSize)
   parent.__init(self)
   self.alpha = alpha or 1
   self.inputSize = inputSize
   self.noise = torch.Tensor()
   self.noise:resizeAs(torch.Tensor(inputSize))
   self.noise:uniform(-alpha, alpha)
end

function Rack:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output = self.output + self.noise:view(1, self.inputSize):expandAs(self.output)
   return self.output
end

function Rack:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end
