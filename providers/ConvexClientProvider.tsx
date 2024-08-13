"use client";

import React, { useEffect } from "react";
import { ClerkProvider, SignIn, useAuth } from "@clerk/nextjs";
import { ConvexProviderWithClerk } from "convex/react-clerk";
import {
  ConvexReactClient,
  Authenticated,
  AuthLoading,
  Unauthenticated,
} from "convex/react";
import LoadingLogo from "@/components/shared/LoadingLogo";

type Props = {
  children: React.ReactNode;
};

const CONVEX_URL = process.env.NEXT_PUBLIC_CONVEX_URL || "";

const convex = new ConvexReactClient(CONVEX_URL);

const RedirectToSignIn = () => {
  useEffect(() => {
    window.location.href = "https://rested-serval-15.accounts.dev/sign-in";
  }, []);

  return null; 
};

const ConvexClientProvider = ({ children }: Props) => {
  return (
    <ClerkProvider>
      <ConvexProviderWithClerk useAuth={useAuth} client={convex}>
        <Unauthenticated>
          {/* <RedirectToSignIn /> */}
          <SignIn/>
        </Unauthenticated>
        <Authenticated>{children}</Authenticated>
        <AuthLoading>
          <LoadingLogo />
        </AuthLoading>
      </ConvexProviderWithClerk>
    </ClerkProvider>
  );
};

export default ConvexClientProvider;
