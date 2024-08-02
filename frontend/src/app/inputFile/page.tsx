import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export function InputFile() {
  return (
    <div className="grid w-full max-w-sm items-center flex-row gap-1.5 ">
      <Label htmlFor="picture">  </Label>
      <Input  id="picture" type="file" />
    </div>
  )
}
