import React,{ ReactNode } from "react"
import Image from 'next/image';

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

// Import the image
import outputImage from './output.png';

interface CardWithFormProps {
  children?: ReactNode;
}

export function CardWithForm() {
  return (
    <Card className="w-[600px] h-[450px] bg-black relative overflow-hidden">
      <div className="relative w-full h-full">
        <Image
          src={outputImage}
          alt="Sample output"
          layout="fill"
          objectFit="cover"
          className="absolute"
        />
      </div>
    </Card>
  )
}
