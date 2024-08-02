"use client"

import * as React from "react"
import Link from "next/link"

import { cn } from "@/lib/utils"
// import { Icons } from "@/components/icons"
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu"
import { ModeToggle } from "./toggle"
import { AccordionDemo } from "./accordion"

const components: { title: string; href: string; description: string }[] = [
  {
    title: "Alert Dialog",
    href: "/docs/primitives/alert-dialog",
    description:
      " ",
  },
  {
    title: "Hover Card",
    href: "/docs/primitives/hover-card",
    description:
      " ",
  },
  {
    title: "Progress",
    href: "/docs/primitives/progress",
    description:
      " ",
  },
  {
    title: "Scroll-area",
    href: "/docs/primitives/scroll-area",
    description: " ",
  },
  {
    title: "Tabs",
    href: "/docs/primitives/tabs",
    description:
      " ",
  },
  {
    title: "Tooltip",
    href: "/docs/primitives/tooltip",
    description:
      " ",
  },
]

export function NavigationMenuDemo() {
  return (

    <NavigationMenu className="ml-20">
      <NavigationMenuList>
        <NavigationMenuItem>
          <NavigationMenuTrigger>Component A</NavigationMenuTrigger>
          <NavigationMenuContent >
            <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
              <li className="row-span-3">
                <NavigationMenuLink asChild>
                  <a
                    className="flex h-full w-full select-none flex-col justify-end rounded-md bg-gradient-to-b from-muted/50 to-muted p-6 no-underline outline-none focus:shadow-md"
                    href="/"
                  >
                    {/* <Icons.logo className="h-6 w-6" /> */}
                    <div className="mb-2 mt-4 text-lg font-medium">
                    
                    </div>
                    <p className="text-sm leading-tight text-muted-foreground"> 
                      
                    </p>
                  </a>
                </NavigationMenuLink>
              </li>
              <ListItem href="/docs" title="A">
                
              </ListItem>
              <ListItem href="/docs/installation" title="B">
                
              </ListItem>
              <ListItem href="/docs/primitives/typography" title="C">
                
              </ListItem>
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>

        <NavigationMenuItem>
          <NavigationMenuTrigger>Component A</NavigationMenuTrigger>
          <NavigationMenuContent >
            <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
              <li className="row-span-3">
                <NavigationMenuLink asChild>
                  <a
                    className="flex h-full w-full select-none flex-col justify-end rounded-md bg-gradient-to-b from-muted/50 to-muted p-6 no-underline outline-none focus:shadow-md"
                    href="/"
                  >
                    {/* <Icons.logo className="h-6 w-6" /> */}
                    <div className="mb-2 mt-4 text-lg font-medium">
                    
                    </div>
                    <p className="text-sm leading-tight text-muted-foreground"> 
                      
                    </p>
                  </a>
                </NavigationMenuLink>
              </li>
              <ListItem href="/docs" title="A">
                
              </ListItem>
              <ListItem href="/docs/installation" title="B">
                
              </ListItem>
              <ListItem href="/docs/primitives/typography" title="C">
                
              </ListItem>
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>

        <NavigationMenuItem>
          <NavigationMenuTrigger>Component A</NavigationMenuTrigger>
          <NavigationMenuContent >
            <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
              <li className="row-span-3">
                <NavigationMenuLink asChild>
                  <a
                    className="flex h-full w-full select-none flex-col justify-end rounded-md bg-gradient-to-b from-muted/50 to-muted p-6 no-underline outline-none focus:shadow-md"
                    href="/"
                  >
                    {/* <Icons.logo className="h-6 w-6" /> */}
                    <div className="mb-2 mt-4 text-lg font-medium">
                    
                    </div>
                    <p className="text-sm leading-tight text-muted-foreground"> 
                      
                    </p>
                  </a>
                </NavigationMenuLink>
              </li>
              <ListItem href="/docs" title="A">
                
              </ListItem>
              <ListItem href="/docs/installation" title="B">
                
              </ListItem>
              <ListItem href="/docs/primitives/typography" title="C">
                
              </ListItem>
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>

        <NavigationMenuItem>
          <NavigationMenuTrigger>Component A</NavigationMenuTrigger>
          <NavigationMenuContent >
            <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
              <li className="row-span-3">
                <NavigationMenuLink asChild>
                  <a
                    className="flex h-full w-full select-none flex-col justify-end rounded-md bg-gradient-to-b from-muted/50 to-muted p-6 no-underline outline-none focus:shadow-md"
                    href="/"
                  >
                    {/* <Icons.logo className="h-6 w-6" /> */}
                    <div className="mb-2 mt-4 text-lg font-medium">
                    
                    </div>
                    <p className="text-sm leading-tight text-muted-foreground"> 
                      
                    </p>
                  </a>
                </NavigationMenuLink>
              </li>
              <ListItem href="/docs" title="A">
                
              </ListItem>
              <ListItem href="/docs/installation" title="B">
                
              </ListItem>
              <ListItem href="/docs/primitives/typography" title="C">
                
              </ListItem>
            </ul>
          </NavigationMenuContent>
        </NavigationMenuItem>

        <NavigationMenuItem>
          <Link href="/docs" legacyBehavior passHref>
            <NavigationMenuLink className={navigationMenuTriggerStyle()}>
              DocS
            </NavigationMenuLink>
          </Link>
        </NavigationMenuItem>
        <ModeToggle/>
      </NavigationMenuList>
    </NavigationMenu>

  )
}

const ListItem = React.forwardRef<
  React.ElementRef<"a">,
  React.ComponentPropsWithoutRef<"a">
>(({ className, title, children, ...props }, ref) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          ref={ref}
          className={cn(
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none">{title}</div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  )
})
ListItem.displayName = "ListItem"
