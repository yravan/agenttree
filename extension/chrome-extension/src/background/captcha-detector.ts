/**
 * CAPTCHA detection module for AgentTree.
 *
 * Runs after page loads to detect common CAPTCHA/anti-bot patterns.
 * Called from the background service worker via chrome.scripting.executeScript.
 */

export const CAPTCHA_SELECTORS = [
  '.g-recaptcha',
  '.h-captcha',
  '[class*="cf-turnstile"]',
  'iframe[src*="recaptcha"]',
  'iframe[src*="hcaptcha"]',
  'iframe[src*="challenges.cloudflare"]',
  '#challenge-running',
  '#challenge-stage',
  '.challenge-running',
];

export const CAPTCHA_TEXT_PATTERNS = [
  /checking your browser/i,
  /access denied/i,
  /please verify you are a human/i,
  /ray id/i,
  /one more step/i,
  /please complete the security check/i,
  /verify you are human/i,
  /just a moment/i,
  /attention required/i,
];

export const ANTIBOT_SCRIPTS = ['dd.js', 'px', 'kasada', 'akamaibmp'];

/**
 * Injected into tabs to check for CAPTCHA patterns.
 * Returns an object with detected patterns.
 */
export function detectCaptchaInPage(): { detected: boolean; patterns: string[] } {
  const patterns: string[] = [];

  // CSS selectors
  const selectors = [
    '.g-recaptcha',
    '.h-captcha',
    '[class*="cf-turnstile"]',
    'iframe[src*="recaptcha"]',
    'iframe[src*="hcaptcha"]',
    'iframe[src*="challenges.cloudflare"]',
    '#challenge-running',
    '#challenge-stage',
    '.challenge-running',
  ];

  for (const selector of selectors) {
    try {
      if (document.querySelector(selector)) {
        patterns.push(`selector:${selector}`);
      }
    } catch {
      // skip invalid selectors
    }
  }

  // Page text
  const bodyText = document.body?.innerText || '';
  const textPatterns = [
    { re: /checking your browser/i, name: 'checking_browser' },
    { re: /access denied/i, name: 'access_denied' },
    { re: /please verify you are a human/i, name: 'verify_human' },
    { re: /ray id/i, name: 'ray_id' },
    { re: /one more step/i, name: 'one_more_step' },
    { re: /verify you are human/i, name: 'verify_human_2' },
    { re: /just a moment/i, name: 'just_a_moment' },
    { re: /attention required/i, name: 'attention_required' },
  ];

  for (const { re, name } of textPatterns) {
    if (re.test(bodyText)) {
      patterns.push(`text:${name}`);
    }
  }

  // Anti-bot scripts
  const scripts = document.querySelectorAll('script[src]');
  const antibotScripts = ['dd.js', 'px', 'kasada', 'akamaibmp'];
  for (const script of scripts) {
    const src = (script as HTMLScriptElement).src.toLowerCase();
    for (const ab of antibotScripts) {
      if (src.includes(ab)) {
        patterns.push(`script:${ab}`);
      }
    }
  }

  return { detected: patterns.length > 0, patterns };
}

/**
 * Setup CAPTCHA detection listener that runs after each navigation.
 */
export function setupCaptchaDetection(
  onDetected: (tabId: number, patterns: string[]) => void,
): void {
  chrome.webNavigation.onCompleted.addListener(async details => {
    if (details.frameId !== 0) return; // Only main frame

    try {
      const [result] = await chrome.scripting.executeScript({
        target: { tabId: details.tabId },
        func: detectCaptchaInPage,
      });

      if (result?.result?.detected) {
        onDetected(details.tabId, result.result.patterns);
      }
    } catch {
      // Tab may have been closed or restricted
    }
  });
}
