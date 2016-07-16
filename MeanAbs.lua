local MeanAbs, parent = torch.class('nn.MeanAbs', 'nn.Module')

function MeanAbs:__init()
   parent.__init(self)
end

function MeanAbs:updateOutput(input)
   input.THNN.Abs_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   self.output = torch.add(self.output, -torch.mean(self.output, 2):expandAs(self.output))
   return self.output
end

function MeanAbs:updateGradInput(input, gradOutput)
   input.THNN.Abs_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata()
   )
   return self.gradInput
end
