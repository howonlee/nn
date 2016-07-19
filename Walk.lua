local Walk, parent = torch.class('nn.Walk', 'nn.Module')

function Walk:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
   self.noise = torch.Tensor()
end

function Walk:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.noise:resizeAs(input)
   self.noise:gaussian(something)
   self.output:add something
   return self.output
end

function Walk:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end
