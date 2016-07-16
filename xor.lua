local xor, parent = torch.class('nn.xor', 'nn.Module')

function xor:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
end

function xor:updateOutput(input)
   input.THNN.xor_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.alpha
   )
   return self.output
end

function xor:updateGradInput(input, gradOutput)
   input.THNN.xor_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.alpha
   )
   return self.gradInput
end
