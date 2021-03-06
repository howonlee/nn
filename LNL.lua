local LNL, parent = torch.class('nn.LNL', 'nn.Module')

function LNL:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
end

function LNL:updateOutput(input)
   input.THNN.LNL_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.alpha
   )
   return self.output
end

function LNL:updateGradInput(input, gradOutput)
   input.THNN.LNL_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.alpha
   )
   return self.gradInput
end
