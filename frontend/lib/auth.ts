/**
 * NextAuth.js v4 Configuration
 *
 * Authenticates against the existing backend JWT auth system at
 * POST /api/v1/auth/token (OAuth2 form-encoded).
 *
 * NEXTAUTH_SECRET must be set in environment.
 * Generate with: openssl rand -base64 32
 */

import type { NextAuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const ACCESS_TOKEN_LEEWAY_MS = 60_000;

type BackendTokens = {
  access_token: string;
  refresh_token: string;
  expires_in: number;
};

async function refreshBackendTokens(refreshToken: string): Promise<BackendTokens | null> {
  try {
    const response = await fetch(`${API_URL}/api/v1/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!response.ok) return null;

    const data = (await response.json()) as Partial<BackendTokens>;
    if (!data.access_token || !data.refresh_token || !data.expires_in) return null;
    return data as BackendTokens;
  } catch {
    return null;
  }
}

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          const body = new URLSearchParams();
          body.append('username', credentials.email);
          body.append('password', credentials.password);
          body.append('grant_type', 'password');

          const response = await fetch(`${API_URL}/api/v1/auth/token`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: body.toString(),
          });

          if (!response.ok) {
            return null;
          }

          const data = await response.json();

          if (!data.access_token) {
            return null;
          }

          // Return a user object that NextAuth will encode into the JWT
          return {
            id: credentials.email,
            email: credentials.email,
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            accessTokenExpiresAt: Date.now() + data.expires_in * 1000,
          };
        } catch {
          return null;
        }
      },
    }),
  ],

  session: {
    strategy: 'jwt',
    // Backend refresh tokens expire after seven days. The cookie lifetime must
    // never outlast that server-side credential.
    maxAge: 7 * 24 * 60 * 60,
    updateAge: 24 * 60 * 60,
  },

  callbacks: {
    async jwt({ token, user }) {
      // On initial sign-in, persist backend tokens into the JWT
      if (user) {
        token.accessToken = (user as { accessToken?: string }).accessToken;
        token.refreshToken = (user as { refreshToken?: string }).refreshToken;
        token.accessTokenExpiresAt = (user as { accessTokenExpiresAt?: number }).accessTokenExpiresAt;
        token.email = user.email;
        return token;
      }

      if (token.accessTokenExpiresAt && Date.now() < (token.accessTokenExpiresAt as number) - ACCESS_TOKEN_LEEWAY_MS) {
        return token;
      }
      if (!token.refreshToken) return { ...token, authError: 'RefreshAccessTokenError' };

      const refreshed = await refreshBackendTokens(token.refreshToken as string);
      if (!refreshed) return { ...token, authError: 'RefreshAccessTokenError' };

      token.accessToken = refreshed.access_token;
      token.refreshToken = refreshed.refresh_token;
      token.accessTokenExpiresAt = Date.now() + refreshed.expires_in * 1000;
      return token;
    },

    async session({ session, token }) {
      // Keep backend bearer tokens inside the encrypted NextAuth cookie.
      (session as { authError?: string }).authError = token.authError as string | undefined;
      if (session.user) {
        session.user.email = token.email as string;
      }
      return session;
    },
  },

  pages: {
    signIn: '/login',
  },

  // NEXTAUTH_SECRET is read automatically from env by NextAuth
  // No need to explicitly set `secret` here if NEXTAUTH_SECRET is defined
};
