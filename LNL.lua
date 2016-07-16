local LNL, parent = torch.class('nn.LNL', 'nn.Module')

function LNL:__init()
   parent.__init(self)
end

function LNL:updateOutput(input)
   input.THNN.LNL_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function LNL:updateGradInput(input, gradOutput)
   input.THNN.LNL_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata()
   )
   return self.gradInput
end
