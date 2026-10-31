const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 990 }, deviceScaleFactor: 1 });
  await page.goto('file://' + src);
  await page.evaluate(() => document.fonts.ready);
  const overflow = await page.evaluate(() => {
    const main = document.querySelector('main');
    const footer = document.querySelector('footer');
    return main.getBoundingClientRect().bottom - footer.getBoundingClientRect().top;
  });
  console.log('content overlaps footer by px:', overflow);
  await page.screenshot({ path: out });
  await browser.close();
})();
