// frontend

"use client";

import React, { useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { Card } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { api } from "@/convex/_generated/api";
import { useConversation } from "@/hooks/useConversation";
import { useMutationState } from "@/hooks/useMutationState";
import { zodResolver } from "@hookform/resolvers/zod";
import { ConvexError } from "convex/values";
import { toast } from "sonner";
import { z } from "zod";
import TextareaAutosize from "react-textarea-autosize";
import { Button } from "@/components/ui/button";
import { SendHorizonal, Mic, Play } from "lucide-react";
import classNames from "classnames"; 

type Props = {};

const chatMessageSchema = z.object({
  content: z.string().min(1, {
    message: "This field can't be empty",
  }),
});

const ChatInput = (props: Props) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [predictedWord, setPredictedWord] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isFetchingPrediction, setIsFetchingPrediction] = useState<boolean>(false);
  const recognitionRef = useRef<any>(null); 

  const { conversationId } = useConversation();

  const { mutate: createMessage, pending } = useMutationState(
    api.message.create
  );

  const form = useForm<z.infer<typeof chatMessageSchema>>({
    resolver: zodResolver(chatMessageSchema),
    defaultValues: {
      content: "",
    },
  });

  useEffect(() => {
    if (!isFetchingPrediction) return;

    const fetchPrediction = async () => {
      try {
        const response = await fetch(
          `http://192.168.40.24:5000/conversations/${conversationId}`,
          {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
          }
        );
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.prediction) {
          setPredictedWord((prevWord) => prevWord + data.prediction);
          form.setValue("content", predictedWord + data.prediction);
        }
      } catch (error) {
        console.error("Error fetching prediction:", error);
      }
    };

    const intervalId = setInterval(fetchPrediction, 3000);

    return () => clearInterval(intervalId);
  }, [conversationId, predictedWord, form, isFetchingPrediction]);

  const handleInputChange = (event: any) => {
    const { value, selectionStart } = event.target;

    if (selectionStart !== null) {
      form.setValue("content", value);
      setPredictedWord(value);
    }
  };

  const handleSubmit = async (values: z.infer<typeof chatMessageSchema>) => {
    createMessage({
      conversationId,
      type: "text",
      content: [values.content],
    })
      .then(() => {
        form.reset();
        setPredictedWord("");
      })
      .catch((error) => {
        toast.error(
          error instanceof ConvexError
            ? error.data
            : "Unexpected error occurred"
        );
      });
  };

  const startRecognition = () => {
    if (!("webkitSpeechRecognition" in window)) {
      toast.error("Your browser does not support speech recognition.");
      return;
    }

    const recognition = new (window as any).webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      const speechResult = event.results[0][0].transcript;
      const newPredictedWord = predictedWord + " " + speechResult;
      setPredictedWord(newPredictedWord);
      form.setValue("content", newPredictedWord);
    };

    recognition.onerror = (event: any) => {
      toast.error("Error occurred in recognition: " + event.error);
      stopRecognition();
    };

    recognition.onend = () => {
      if (isRecording) {
        recognition.start(); 
      }
    };

    recognitionRef.current = recognition;
    recognition.start(); 
  };

  const stopRecognition = () => {
    setIsRecording(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop(); 
      recognitionRef.current = null; 
    }
  };

  const handleMicPress = () => {
    setIsRecording(true);
    startRecognition();
  };

  const handleMicRelease = () => {
    stopRecognition();
  };

  const togglePredictionFetching = () => {
    setIsFetchingPrediction(!isFetchingPrediction);
    toast.success(isFetchingPrediction ? "Gloves mode deactivated" : "Gloves mode activated");
  };

  return (
    <Card className="w-full p-2 rounded-lg relative">
      <div className="flex gap-2 items-end w-full">
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleSubmit)}
            className="flex gap-2 items-end w-full"
          >
            <FormField
              control={form.control}
              name="content"
              render={({ field }) => {
                return (
                  <FormItem className="h-full w-full">
                    <FormControl>
                      <TextareaAutosize
                        onKeyDown={async (e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            await form.handleSubmit(handleSubmit)();
                          }
                        }}
                        rows={1}
                        maxRows={3}
                        {...field}
                        onChange={handleInputChange}
                        onClick={handleInputChange}
                        placeholder="Type a message..."
                        className="min-h-full w-full resize-none border-0 outline-0 bg-card text-card-foreground placeholder:text-muted-foreground p-1.5"
                        value={predictedWord}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                );
              }}
            />
            <Button
              onMouseDown={handleMicPress}
              onMouseUp={handleMicRelease}
              onTouchStart={handleMicPress}
              onTouchEnd={handleMicRelease}
              size="icon"
              type="button"
              className={classNames("transition-transform duration-200", {
                "scale-75": isRecording,
              })}
            >
              <Mic />
            </Button>

            <Button
              onClick={togglePredictionFetching}
              size="icon"
              type="button"
              className={classNames("transition-transform duration-200", {
                "scale-75": isFetchingPrediction,
              })}
            >
              <Play />
            </Button>

            <Button disabled={pending} size="icon" type="submit">
              <SendHorizonal />
            </Button>
          </form>
        </Form>
      </div>
    </Card>
  );
};

export default ChatInput;