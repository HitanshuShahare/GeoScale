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

type Status = {
  value: string
  label: string
}

const statuses: Status[] = [
  {
    value: "Inverse Distance Weighting (IDW)",
    label: "Inverse Distance Weighting (IDW)",
  },
  {
    value: "Kriging",
    label: "Kriging",
  },
  {
    value: "Radial Basis Function (RBF)",
    label: "Radial Basis Function (RBF)",
  },
  {
    value: "Spline Interpolation",
    label: "Spline Interpolation",
  },
  {
    value: "Natural Neighbor Interpolation",
    label: "Natural Neighbor Interpolation",
  },
  {
    value: "Trend Surface Analysis",
    label: "Trend Surface Analysis",
  },
  {
    value: "Local Polynomial Interpolation",
    label: "Local Polynomial Interpolation",
  },
  {
    value: "Co-Kriging",
    label: "Co-Kriging",
  },
  {
    value: "Multiple-Point Geostatistics",
    label: "Multiple-Point Geostatistics",
  },
  {
    value: "Semi-Variogram Interpolation",
    label: "Semi-Variogram Interpolation",
  },
  {
    value: "Triangulated Irregular Network (TIN) Interpolation",
    label: "Triangulated Irregular Network (TIN) Interpolation",
  },
  {
    value: "Shepard's Method",
    label: "Shepard's Method",
  },
  {
    value: "Voronoi Diagram",
    label: "Voronoi Diagram",
  },
  {
    value: "Nearest Neighbor Interpolation",
    label: "Nearest Neighbor Interpolation",
  },
  {
    value: "Global Polynomial Interpolation",
    label: "Global Polynomial Interpolation",
  },
]

export function ComboboxPopover() {
  const [open, setOpen] = React.useState(false)
  const [selectedStatus, setSelectedStatus] = React.useState<Status | null>(
    null
  )

  return (
    <div className="flex items-center space-x-4">
      <p className="text-sm text-muted-foreground"><b> METHODS</b></p>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-[550px] justify-start">
            {selectedStatus ? <>{selectedStatus.label}</> : <>+ Select</>}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="p-0" side="right" align="start">
          <Command>
            <CommandInput placeholder="Change status..." />
            <CommandList>
              <CommandEmpty>No results found.</CommandEmpty>
              <CommandGroup>
                {statuses.map((status) => (
                  <CommandItem
                    key={status.value}
                    value={status.value}
                    onSelect={(value) => {
                      setSelectedStatus(
                        statuses.find((priority) => priority.value === value) ||
                          null
                      )
                      setOpen(false)
                    }}
                  >
                    {status.label}
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
