"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

// Define the Status type
type Status = {
  value: string
  label: string
}

// Define the ComboboxPopover2 component
export function ComboboxPopover3({ data }: { data: Status[] }) {
  const [open, setOpen] = React.useState(false)
  const [selectedData, setSelectedData] = React.useState<Status | null>(null)

  return (
    <div className="flex items-center space-x-4">
      <p className="text-sm text-muted-foreground"><b> Select Type </b></p>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-[550px] justify-start">
            {selectedData ? <>{selectedData.label}</> : <>+ Select type</>}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="p-0" side="right" align="start">
          <Command>
            <CommandInput placeholder="Change data..." />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup>
                {data.map((item) => (
                  <CommandItem
                    key={item.value}
                    value={item.value}
                    onSelect={(value) => {
                      setSelectedData(
                        data.find((d) => d.value === value) || null
                      )
                      setOpen(false)
                    }}
                  >
                    {item.label}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  )
}

